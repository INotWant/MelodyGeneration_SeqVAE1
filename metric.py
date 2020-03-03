"""
旋律的编码
```
关于 Melody 的 90 维编码的意义
 0 维表示“持续”；
 1 维表示“休息”；
 其他维度表示相应的音高；
    21-108
```

根据旋律的编码实现计算 EB、UPC、QN
"""


def EB(musics):
    """
    EB = 空小节的数目 / 总小节数
    :return 平均 EB
    """
    bar_num = 256 // 16
    empty_bar_num = 0
    for music in musics:  # music --> 256 * 90
        for i in range(bar_num):
            is_empty_bar = True
            for j in range(16):
                if music[i * 16 + j][0] == 0. and music[i * 16 + j][1] == 0.:
                    is_empty_bar = False
                    break
            if is_empty_bar:
                empty_bar_num += 1
    return empty_bar_num / (bar_num * len(musics))


def UPC_for_music(music):
    """
    UPC = 1/n * (x_1+...+x_n) ，x_i 表示第i小节所含的音高种类数
    :return 根据单一样本求得的 UPC
    """
    bar_num = 256 // 16
    pitch_class_num = 0
    for i in range(bar_num):
        pitch_class = set()
        for j in range(16):
            if music[i * 16 + j][0] == 1.0 or music[i * 16 + j][1] == 1.0:
                continue
            for k in range(2, 90):
                if music[i * 16 + j][k] == 1.0:
                    pitch_class.add((k + 19) / 12)
                    break
        pitch_class_num += len(pitch_class)
    return pitch_class_num / bar_num


def UPC(musics):
    """ UPC
    :return 平均 UPC
    """
    sum = 0
    count = 0
    for music in musics:
        sum += UPC_for_music(music)
        count += 1
    return sum / count


def UPN_for_music(music):
    bar_num = 256 // 16
    pitch_num = 0
    for i in range(bar_num):
        for j in range(16):
            if music[i * 16 + j][0] == 1.0 or music[i * 16 + j][1] == 1.0:
                continue
            pitch_num += 1
    return pitch_num / bar_num


def UPN(musics):
    sum = 0
    count = 0
    for music in musics:
        sum += UPN_for_music(music)
        count += 1
    return sum / count


def QN_for_music(music, qualified_time_step):
    """
    QN = “合格”音符的数目 / （一段音乐的）音符总数
    """
    total_time_step = 256
    note_total_num = 0
    note_qualified_num = 0
    note_continue_ts = 0
    for i in range(total_time_step):
        k = 0
        while k < 90:
            if music[i][k] == 1.0:
                break
            k += 1
        if k == 0:
            if note_continue_ts != 0:
                note_continue_ts += 1
        elif k == 1:
            if note_continue_ts >= qualified_time_step:
                note_qualified_num += 1
            note_continue_ts = 0
        else:
            note_total_num += 1
            if note_continue_ts >= qualified_time_step:
                note_qualified_num += 1
            note_continue_ts = 1
    if note_continue_ts >= qualified_time_step:
        note_qualified_num += 1
    if note_total_num == 0:
        return 0
    return note_qualified_num / note_total_num


def QN(musics, qualified_time_step):
    """ QN
    :return 平均 QN
    """
    count = 0
    sum = 0
    for music in musics:
        sum += QN_for_music(music, qualified_time_step)
        count += 1
    return sum / count


def SPB_for_music(music):
    bar_num = 256 // 16
    pitch_class_num = 0
    for i in range(bar_num):
        interval_min = 10
        interval_max = -1
        for j in range(16):
            if music[i * 16 + j][0] == 1.0 or music[i * 16 + j][1] == 1.0:
                continue
            for k in range(2, 90):
                if music[i * 16 + j][k] == 1.0:
                    interval = (k + 19) / 12
                    if interval > interval_max:
                        interval_max = interval
                    if interval < interval_min:
                        interval_min = interval
                    break
        pitch_class_num += max((interval_max - interval_min), 0)
    return pitch_class_num / bar_num


def SPB(musics):
    sum = 0
    count = 0
    for music in musics:
        sum += SPB_for_music(music)
        count += 1
    return sum / count


import os

import numpy as np

if __name__ == '__main__':
    path = 'output/create'
    fs = os.listdir(path)
    fs.sort()

    for fn in fs:
        print('===', fn)
        musics = np.load(os.path.join(path, fn))
        eq = EB(musics)
        upc = UPC(musics)
        upn = UPN(musics)
        qn2 = QN(musics, qualified_time_step=2)
        qn4 = QN(musics, qualified_time_step=4)
        qn6 = QN(musics, qualified_time_step=6)
        qn8 = QN(musics, qualified_time_step=8)
        spb = SPB(musics)

        print('%.2f%% %.2f %.2f %.2f%% %.2f%% %.2f%% %.2f%% %.2f' % (
            eq * 100, upc, upn, qn2 * 100, qn4 * 100, qn6 * 100, qn8 * 100, spb))
