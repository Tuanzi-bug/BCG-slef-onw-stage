def Z_ScoreNormalization(x):
    import numpy as np
    out = (x - np.mean(x)) / np.std(x)
    return out
def get_peaks_bestParameters(input, best_distance=1.0, best_prominence=0.3, best_rate=1):
    """
    Find peaks in the input signal with simplified parameters.

    Parameters:
    - input: array-like
        The input signal to process.
    - distance_factor: float, optional
        Factor to adjust the calculated distance between peaks. Default is 0.8.
    - initial_prominence: float, optional
        The prominence threshold for the first pass of peak filtering. Default is 0.35.
    - final_prominence: float, optional
        The prominence threshold for the final peak filtering. Default is 0.5.

    Returns:
    - pppeaks: array-like
        The indices of the final detected peaks.
    """
    from scipy import signal
    import numpy as np
    input = Z_ScoreNormalization(input)
    # 获取所有的峰值点
    peaks = signal.argrelextrema(input, np.greater)[0]
    peaks_h = input[peaks]
    peaks_h = Z_ScoreNormalization(peaks_h)
    # 根据峰值点的变化进行筛选
    pp = signal.find_peaks(peaks_h,distance=best_distance,height=0,prominence=best_prominence)
    pp = pp[0]
    ppeaks = peaks[pp]
    # print(int(len(ppeaks)), int(6000 / int(len(ppeaks))))
    # 根据大概的心跳点间隔进行寻找峰值点
    pppeaks = signal.find_peaks(input, height=0, distance=40)[0]
    # pppeaks = delete_close(pppeaks, input)
    return pppeaks