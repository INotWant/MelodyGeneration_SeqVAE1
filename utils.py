import numpy as np
import pretty_midi
import tensorflow as tf

import data


def flatten_maybe_padded_sequences(maybe_padded_sequences, lengths=None):
    """Flattens the batch of sequences, removing padding (if applicable).

    Args:
      maybe_padded_sequences: A tensor of possibly padded sequences to flatten,
          sized `[N, M, ...]` where M = max(lengths).
      lengths: Optional length of each sequence, sized `[N]`. If None, assumes no
          padding.

    Returns:
       flatten_maybe_padded_sequences: The flattened sequence tensor, sized
           `[sum(lengths), ...]`.
    """

    def flatten_unpadded_sequences():
        # The sequences are equal length, so we should just flatten over the first
        # two dimensions.
        return tf.reshape(maybe_padded_sequences,
                          [-1] + maybe_padded_sequences.shape.as_list()[2:])

    if lengths is None:
        return flatten_unpadded_sequences()

    def flatten_padded_sequences():
        indices = tf.where(tf.sequence_mask(lengths))
        return tf.gather_nd(maybe_padded_sequences, indices)

    return tf.cond(
        tf.equal(tf.reduce_min(lengths), tf.shape(maybe_padded_sequences)[1]),
        flatten_unpadded_sequences,
        flatten_padded_sequences)


def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.

    see: https://github.com/LantaoYu/SeqGAN/blob/master/discriminator.py

    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(tf.layers.dense(input_,
                                  size,
                                  name='highway_lin_%d' % idx,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.05)))

            t = tf.sigmoid(tf.layers.dense(input_,
                                           size,
                                           name='highway_gate_%d' % idx,
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.05))
                           + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


def data_confusion(batch_size, fake_data, true_data):
    """分别从 fake_data true_data 中拿出一半的数据组建成新数据集，并为新数据添加标签。最后对新数据集进行 shuffle

    :param batch_size:
    :param fake_data:
    :param true_data:
    :return: a tuple of tensor op, (data_inputs, data_labels)
    """
    num = batch_size // 2
    data_inputs = tf.concat([fake_data[:num], true_data[:num]], axis=0)

    fake_labels = [[0, 1] for _ in range(num)]
    true_labels = [[1, 0] for _ in range(num)]
    data_labels = tf.convert_to_tensor(np.concatenate((fake_labels, true_labels), 0))
    data_labels = tf.to_float(data_labels)
    data_labels = tf.tile(tf.expand_dims(data_labels, axis=1), [1, tf.shape(data_inputs)[1], 1])

    data_all = tf.concat([data_inputs, data_labels], axis=-1)
    # shuffle
    # https://github.com/tensorflow/tensorflow/issues/6269
    data_all = tf.gather(data_all, tf.random.shuffle(tf.range(tf.shape(data_all)[0])))

    data_inputs_op, data_labels = tf.split(data_all, [-1, 2], axis=2)
    data_labels = data_labels[:, :1, :]
    data_labels_op = tf.squeeze(data_labels, axis=1)

    return data_inputs_op, data_labels_op


def dataset_fn(config, num_threads=4, is_training=True, cache_dataset=True):
    """对 Magenta 项目脚本（通过编码MIDI文件）生成的 tfrecord 文件读取的预处理
    Note：本函数是对 data 的封装，需要 config 具备一系列属性以满足 data 中的要求
    """
    return data.get_dataset(
        config,
        tf_file_reader=tf.data.TFRecordDataset,
        num_threads=num_threads,
        is_training=is_training,
        cache_dataset=cache_dataset)


def get_input_tensors_from_dataset(dataset, batch_size, max_seq_len, pitch_num, control_num=0):
    """从 Magenta 项目脚本（通过编码MIDI文件）生成的 tfrecord 文件读取数据

    :param dataset: 一般为上述 dataset_fn() 的输出
    :param batch_size: batch size
    :param max_seq_len: 序列的最大长度
    :param pitch_num: 音高的维数
    :param control_num: 一般为和弦的维数
    :return: （input_sequence, output_sequence, control_sequence, sequence_length）
    """
    iterator = dataset.make_one_shot_iterator()
    input_sequence, output_sequence, control_sequence, sequence_length = iterator.get_next()

    input_sequence.set_shape([batch_size, max_seq_len, pitch_num])
    input_sequence = tf.to_float(input_sequence)

    output_sequence.set_shape([batch_size, max_seq_len, pitch_num])
    output_sequence = tf.to_float(output_sequence)

    sequence_length.set_shape([batch_size] + sequence_length.shape[1:].as_list())
    sequence_length = tf.minimum(sequence_length, max_seq_len)

    if control_num != 0:
        control_sequence.set_shape([batch_size, max_seq_len, control_num])
        control_sequence = tf.to_float(control_sequence)
    else:
        control_sequence = None
    return input_sequence, output_sequence, control_sequence, sequence_length


def count_all_trainable_variables():
    """计算所有的可训练的参数个数"""
    from functools import reduce
    from operator import mul
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params


def count_all_variables():
    """计算所有变量的参数个数"""
    from functools import reduce
    from operator import mul
    num_params = 0
    for variable in tf.all_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params


# Should not be called from within the graph to avoid redundant summaries.
def trial_summary(hparams, output_dir):
    """Writes a tensorboard text summary of the trial."""
    hparams_dict = hparams.values()

    # Create a markdown table from hparams.
    header = '| Key | Value |\n| :--- | :--- |\n'
    keys = sorted(hparams_dict.keys())
    lines = ['| %s | %s |' % (key, str(hparams_dict[key])) for key in keys]
    hparams_table = header + '\n'.join(lines) + '\n'

    hparam_summary = tf.summary.text(
        'hparams', tf.constant(hparams_table, name='hparams'), collections=[])

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(output_dir, graph=sess.graph)
        writer.add_summary(hparam_summary.eval())
        writer.close()


def slerp(p0, p1, t):
    """Spherical linear interpolation.
    https://blog.csdn.net/u012947821/article/details/17136443
    """
    omega = np.arccos(
        np.dot(np.squeeze(p0 / np.linalg.norm(p0)),
               np.squeeze(p1 / np.linalg.norm(p1))))
    so = np.sin(omega)
    return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1
