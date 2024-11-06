# بسم الله الرحمن الرحيم
"""
Calculating the Last Third of the Night
"""

magrib_t = [6, 2]  # [hr, min]
fajer_t = [5, 47]  # [hr, min]


def last_third(magrib, fajer):
    mag_hr, mag_min = magrib[0], magrib[1]
    faj_hr, faj_min = fajer[0], fajer[1]
    night_hr = (mag_hr + 12) - faj_hr
    night_min = mag_min - faj_min
    night = night_hr + night_min/60
    third = (mag_hr + mag_min/60) + 2/3 * night
    if third > 12:
        third -= 12
    third_dec = third - int(third)
    third_dec *= 60
    if third_dec >= 60:
        third_dec -= 60
        third += 1
    print(f"Last Third of the Night = {int(third)}:{round(third_dec)} AM")


last_third(magrib_t, fajer_t)
