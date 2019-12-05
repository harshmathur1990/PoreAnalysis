import os
import sys
import datetime
import pywt
from datetime import timedelta
from ast import literal_eval as make_tuple
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import segmentation_pyx as seg
from utils import Base, engine
import model
import matplotlib.cm as cm
import sunpy.time
import numpy as np
from utils import get_julian_time
import scipy.interpolate
from waveletFunctions import wavelet, wave_signif
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker


normal_field_list = [
    'eccentricity', 'size', 'mean_intensity',
    'major_axis_length', 'minor_axis_length',
    'orientation', 'qs_intensity', 'qs_contrast'
]


k_list = [
    -3.11,
    -3.08,
    -3.14,
    -3.17,
    -3.20,
    -3.26
]


def plot_according_to_library(field):
    cumulative_differences, interpolated_data = get_interpolated_data(field)

    sst = interpolated_data

    sst = sst - np.mean(sst)

    variance = np.std(sst, ddof=1) ** 2

    n = len(sst)
    dt = 11
    time = np.arange(len(sst)) * dt + 1  # construct time array

    xlim = ([time[0], time[-1]])  # plotting range

    pad = 1  # pad the time series with zeroes (recommended)
    dj = 0.25 / 4  # this will do 4 sub-octaves per octave
    s0 = 2 * dt  # this says start at a scale of 6 months
    j1 = 7 / dj  # this says do 7 powers-of-two with dj sub-octaves each
    lag1 = 0.72  # lag-1 autocorrelation for red noise background
    mother = 'MORLET'

    # Wavelet transform:
    wave, period, scale, coi = wavelet(sst, dt, pad, dj, s0, j1, mother)
    power = (np.abs(wave)) ** 2  # compute wavelet power spectrum
    global_ws = (np.sum(power, axis=1) / n)  # time-average over all times

    # Significance levels:
    signif = wave_signif(
        (
            [variance]
        ),
        dt=dt,
        sigtest=0, scale=scale,
        lag1=lag1, mother=mother
    )
    sig95 = signif[:, np.newaxis].dot(
        np.ones(n)[np.newaxis, :]
    )  # expand signif --> (J+1)x(N) array
    sig95 = power / sig95  # where ratio > 1, power is significant

    # Global wavelet spectrum & significance levels:
    dof = n - scale  # the -scale corrects for padding at edges
    global_signif = wave_signif(
        variance, dt=dt, scale=scale, sigtest=1,
        lag1=lag1, dof=dof, mother=mother
    )

    fig = plt.figure(figsize=(9, 10))

    gs = GridSpec(3, 4, hspace=0.4, wspace=0.75)

    plt.subplots_adjust(
        left=0.1, bottom=0.05,
        right=0.9, top=0.95, wspace=0, hspace=0
    )
    plt.subplot(gs[0, 0:3])
    plt.plot(time, sst, 'k')
    plt.xlim(xlim[:])
    plt.xlabel('Time (Seconds)')
    plt.ylabel('{}'.format(snake_to_camel(field)))
    plt.title('a) {} vs Time'.format(snake_to_camel(field)))

    # --- Contour plot wavelet power spectrum
    # plt3 = plt.subplot(3, 1, 2)
    plt3 = plt.subplot(gs[1, 0:3])
    levels = [0, 0.5, 1, 2, 4, 999]
    CS = plt.contourf(
        time, period, power, len(levels)
    )  # *** or use 'contour'
    im = plt.contourf(
        CS, levels=levels,
        colors=['white', 'bisque', 'orange', 'orangered', 'darkred']
    )
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Period (Seconds)')
    plt.title('b) Wavelet Power Spectrum (contours at 0.5,1,2,4\u00B0C$^2$)')
    plt.xlim(xlim[:])
    # 95# significance contour, levels at -99 (fake) and 1 (95# signif)
    plt.contour(time, period, sig95, [-99, 1], colors='k')
    # cone-of-influence, anything "below" is dubious
    plt.plot(time, coi, 'k')
    # format y-scale
    plt3.set_yscale('log', basey=2, subsy=None)
    plt.ylim([np.min(period), np.max(period)])
    ax = plt.gca().yaxis
    ax.set_major_formatter(ticker.ScalarFormatter())
    plt3.ticklabel_format(axis='y', style='plain')
    plt3.invert_yaxis()
    # set up the size and location of the colorbar

    # position = fig.add_axes([0.5, 0.36, 0.2, 0.01])

    # plt.colorbar(
    #     im, cax=position, orientation='horizontal'
    # )  # fraction=0.05, pad=0.5)

    plt.subplots_adjust(right=0.7, top=0.9)

    # --- Plot global wavelet spectrum
    plt4 = plt.subplot(gs[1, -1])
    plt.plot(global_ws, period)
    plt.plot(global_signif, period, '--')
    plt.xlabel('Power (\u00B0C$^2$)')
    plt.title('c) Global Wavelet Spectrum')
    plt.xlim([0, 1.25 * np.max(global_ws)])
    # format y-scale
    plt4.set_yscale('log', basey=2, subsy=None)
    plt.ylim([np.min(period), np.max(period)])
    ax = plt.gca().yaxis
    ax.set_major_formatter(ticker.ScalarFormatter())
    plt4.ticklabel_format(axis='y', style='plain')
    plt4.invert_yaxis()

    # fig.tight_layout()

    plt.show()
    # plt.savefig(
    #     '{}_contour.png'.format(field),
    #     format='png',
    #     dpi=300,
    #     bbox_inches='tight'
    # )


def get_interpolated_data(field):
    all_records = model.Record.get_all()
    date_list = list()
    field_list = list()

    for a_record in all_records:
        date_list.append(a_record.date_time)
        field_list.append(getattr(a_record, 'mean_' + field))

    julian_dates = get_julian_time(date_list)

    differences = list()

    for index, jd in enumerate(julian_dates):
        if index == 0:
            differences.append(0.0)
        else:
            differences.append(
                (jd - julian_dates[index - 1]) * 86400
            )

    differences = np.round(np.array(differences))

    culprit_index = int(np.where(differences == 504)[0][0])

    new_differences = np.array(
        list(differences[0:culprit_index]) +
        [11] * 45 + [9] +
        list(differences[culprit_index + 1:])
    )

    missing_elements = np.array(
        field_list[0:culprit_index] +
        [np.nan] * 46 +
        field_list[culprit_index + 1:]
    )

    mask = np.isfinite(missing_elements)

    new_differences += 1

    cumulative_differences = np.cumsum(new_differences)

    interpolation_func = scipy.interpolate.interp1d(
        cumulative_differences[mask],
        missing_elements[mask],
        kind='cubic'
    )

    interpolated_data = interpolation_func(cumulative_differences)

    return cumulative_differences, interpolated_data


def get_cwt(field, sampling_interval=11):
    cumulative_differences, interpolated_data = get_interpolated_data(field)

    coefs, freq = pywt.cwt(
        data=interpolated_data,
        scales=np.arange(1, 128),
        wavelet='morl',
        sampling_period=sampling_interval
    )

    return coefs, freq, cumulative_differences, interpolated_data


def save_wavelet_plot(field):

    fig, axs = plt.subplots(2)

    coefs, freq, cumulative_differences, interpolated_data = get_cwt(field)

    coefs = np.abs(coefs) ** 2

    period = 1 / freq

    def f(x, y):
        return coefs[x][y]

    vec_f = np.vectorize(f)

    x = np.arange(coefs.shape[0])

    y = np.arange(coefs.shape[1])

    X, Y = np.meshgrid(x, y)

    Z = vec_f(X, Y)

    axs[0].plot(cumulative_differences, interpolated_data)

    axs[0].set_xlabel('Time in Seconds')

    axs[0].set_ylabel('{}'.format(snake_to_camel(field)))

    axs[0].set_title('{} vs Time Plot'.format(snake_to_camel(field)))

    im = axs[1].contourf(Y, X, Z)

    pos_y = np.int64(np.linspace(0, 126, 10))

    pos_x = np.int64(np.linspace(0, 350, 10))

    yticks = np.round(period[pos_y], decimals=2)

    xticks = cumulative_differences[pos_x]

    axs[1].set_xticks(pos_x)

    axs[1].set_xticklabels(xticks)

    axs[1].set_yticks(pos_y)

    axs[1].set_yticklabels(yticks)

    axs[1].set_xlabel('Time in Seconds')

    axs[1].set_ylabel('Period in Seconds')

    axs[1].set_title('{} Time frequency Plot'.format(snake_to_camel(field)))

    fig.colorbar(im, ax=axs[1])

    fig.tight_layout()

    plt.xticks(rotation=45)

    plt.legend()

    plt.savefig(
        '{}_contour.png'.format(field),
        format='png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.clf()

    plt.cla()


def save_normal_fields_wavelet_plots():
    for field in normal_field_list:
        save_wavelet_plot(field)


def populate_derived_fields():

    all_records = model.Record.get_all()

    for a_record in all_records:

        poredata_list = a_record.poredata

        eccentricity_list = list()

        size_list = list()

        mean_intensity_list = list()

        min_intensity_list = list()

        major_axis_length_list = list()

        minor_axis_length_list = list()

        orientation_list = list()

        centroid_list = list()

        for poredata in poredata_list:
            eccentricity_list.append(poredata.eccentricity)
            size_list.append(poredata.size)
            mean_intensity_list.append(poredata.mean_intensity)
            min_intensity_list.append(poredata.min_intensity)
            major_axis_length_list.append(poredata.major_axis_length)
            minor_axis_length_list.append(poredata.minor_axis_length)
            orientation_list.append(poredata.orientation)
            centroid_list.append(
                list(
                    make_tuple(poredata.centroid)
                )
            )

        a_record.mean_eccentricity = np.mean(eccentricity_list)
        a_record.std_eccentricity = np.std(eccentricity_list)

        a_record.mean_size = np.mean(size_list)
        a_record.std_size = np.std(size_list)

        a_record.mean_mean_intensity = np.mean(mean_intensity_list)
        a_record.std_mean_intensity = np.std(mean_intensity_list)

        a_record.mean_min_intensity = np.mean(min_intensity_list)
        a_record.std_min_intensity = np.std(min_intensity_list)

        a_record.mean_orientation = np.mean(orientation_list)
        a_record.std_orientation = np.std(orientation_list)

        a_record.mean_major_axis_length = np.mean(major_axis_length_list)
        a_record.std_major_axis_length = np.std(major_axis_length_list)

        a_record.mean_minor_axis_length = np.mean(minor_axis_length_list)
        a_record.std_minor_axis_length = np.std(minor_axis_length_list)

        a_record.mean_centroid = str(
            tuple(
                np.mean(
                    centroid_list, axis=0
                )
            )
        )
        a_record.std_centroid = str(
            tuple(
                np.std(
                    centroid_list, axis=0
                )
            )
        )

        a_record.save()


def snake_to_camel(word):
        return ' '.join(x.capitalize() or '_' for x in word.split('_'))


def get_error_scatter_plots(field1, field2):
    all_records = model.Record.get_all()

    date_list = list()

    time_in_sec = list()

    x_list = list()

    y_list = list()

    xerr = list()

    yerr = list()

    for a_record in all_records:
        date_list.append(a_record.date_time)

        if isinstance(field2, tuple):
            x_list.append(
                list(
                    make_tuple(
                        getattr(
                            a_record,
                            'mean_' + field2[0]
                        )
                    )
                )[field2[1]]
            )

            xerr.append(
                list(
                    make_tuple(
                        getattr(
                            a_record,
                            'std_' + field2[0]
                        )
                    )
                )[field2[1]]
            )

        else:
            x_list.append(
                getattr(a_record, 'mean_' + field2)
            )

            xerr.append(
                getattr(a_record, 'std_' + field2)
            )

        if isinstance(field1, tuple):
            y_list.append(
                list(
                    make_tuple(
                        getattr(
                            a_record,
                            'mean_' + field1[0]
                        )
                    )
                )[field1[1]]
            )

            yerr.append(
                list(
                    make_tuple(
                        getattr(
                            a_record,
                            'std_' + field1[0]
                        )
                    )
                )[field1[1]]
            )
        else:
            y_list.append(
                getattr(a_record, 'mean_' + field1)
            )

            yerr.append(
                getattr(a_record, 'std_' + field1)
            )

        time_in_sec.append(
            int(
                sunpy.time.parse_time(
                    a_record.date_time.isoformat()
                ).jd * 86400
            )
        )

    first_time_in_sec = time_in_sec[0:76]

    first_time_in_sec -= np.min(first_time_in_sec)

    first_time_in_sec = first_time_in_sec / np.max(first_time_in_sec)

    first_time_in_sec *= 256

    first_time_in_sec = np.int64(first_time_in_sec)

    first_colormap = cm.rainbow(
        np.arange(
            np.max(first_time_in_sec) + 1
        )
    )

    last_time_in_sec = time_in_sec[76:]

    last_time_in_sec -= np.min(last_time_in_sec)

    last_time_in_sec = last_time_in_sec / np.max(last_time_in_sec)

    last_time_in_sec *= 256

    last_time_in_sec = np.int64(last_time_in_sec)

    last_colormap = cm.rainbow(
        np.arange(
            np.max(last_time_in_sec) + 1
        )
    )

    # plt.scatter(
    #     x_list,
    #     y_list,
    #     c=colormap[time_in_sec]
    # )

    fig = plt.figure()

    plt.scatter(
        x_list[0:76],
        y_list[0:76],
        # yerr=yerr[0:76],
        # xerr=xerr[0:76],
        marker='o',
        c=first_colormap[first_time_in_sec]
    )

    plt.scatter(
        x_list[76:],
        y_list[76:],
        # yerr=yerr[76:],
        # xerr=xerr[76:],
        marker='*',
        c=last_colormap[last_time_in_sec]
    )

    if isinstance(field1, tuple):
        plt.title(
            '{} {} vs {} Scatter Plot'.format(
                snake_to_camel(field1[0]),
                snake_to_camel(field1[2]),
                snake_to_camel(field2[2])
            )
        )
        plt.xlabel('{}'.format(snake_to_camel(field2[2])))
        plt.ylabel('{}'.format(snake_to_camel(field1[2])))
    else:
        plt.title(
            '{} vs {} Scatter Plot'.format(
                snake_to_camel(field1), snake_to_camel(field2)
            )
        )
        plt.xlabel('{}'.format(snake_to_camel(field2)))
        plt.ylabel('{}'.format(snake_to_camel(field1)))

    plt.legend()

    fig.tight_layout()

    # annot = plt.annotate(
    #     "",
    #     xy=(0, 0),
    #     xytext=(20, 20),
    #     textcoords="offset points",
    #     bbox=dict(boxstyle="round", fc="w"),
    #     arrowprops=dict(arrowstyle="->")
    # )

    plt.show()

    if isinstance(field1, tuple):

        fig.savefig(
            '{}_{}_vs_{}_scatter.png'.format(field1[0], field1[2], field2[2]),
            format='png',
            dpi=300
        )
    else:
        fig.savefig(
            '{}_vs_{}_scatter.png'.format(field1, field2),
            format='png',
            dpi=300
        )
    plt.clf()
    plt.cla()


def error_plot_field_vs_date(field, x_y=0):
    all_records = model.Record.get_all()

    date_list = list()

    value_list = list()

    yerr = list()

    for a_record in all_records:

        date_list.append(a_record.date_time)

        if field == 'centroid':
            value = make_tuple(getattr(a_record, 'mean_' + field))[x_y]
            value_list.append(
                value
            )
            valueerr = make_tuple(getattr(a_record, 'std_' + field))[x_y]
            yerr.append(
                valueerr
            )
        else:
            if not field.startswith('qs'):
                value_list.append(
                    getattr(a_record, 'mean_' + field)
                )
                yerr.append(
                    getattr(a_record, 'std_' + field)
                )
            else:
                if field != 'qs_contrast':
                    value_list.append(
                        getattr(a_record, field)
                    )
                    yerr.append(
                        getattr(a_record, 'qs_std')
                    )
                else:
                    value_list.append(
                        getattr(a_record, field)
                    )

    fig = plt.figure()

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    plt.gca().xaxis.set_major_locator(mdates.DayLocator())

    _start_date = datetime.datetime(2017, 9, 28, 8, 51, 20)

    x_ticks = list()

    for i in range(0, 9):
        x_ticks.append(
            _start_date + timedelta(minutes=8 * i)
        )

    if len(yerr) != 0:
        plt.errorbar(
            date_list, value_list, yerr=yerr, fmt='b', ecolor='yellow'
        )
    else:
        plt.plot(date_list, value_list)

    if field != 'centroid':
        plt.title('{} vs Time Plot'.format(snake_to_camel(field)))
    else:
        coord = 'X' if x_y == 0 else 'Y'
        plt.title('{} {} coordinate vs Time Plot'.format(
            snake_to_camel(field), coord)
        )
    plt.xlabel('Time')
    plt.ylabel(snake_to_camel(field))
    plt.xticks(x_ticks, rotation=45)
    plt.gcf().autofmt_xdate()
    plt.legend()
    fig.tight_layout()
    plt.show()
    if field != 'centroid':
        plt.savefig('{}_vs_time.png'.format(field), format='png', dpi=300)
    else:
        plt.savefig(
            '{}_{}_vs_time.png'.format(field, coord), format='png', dpi=300
        )
    plt.clf()
    plt.cla()


def save_all_plots():
    # normal_field_list = [
    #     'eccentricity', 'size', 'mean_intensity',
    #     'major_axis_length', 'minor_axis_length',
    #     'orientation'
    # ]

    # for a_field in normal_field_list:
        # error_plot_field_vs_date(a_field)

    # error_plot_field_vs_date('centroid', 0)
    # error_plot_field_vs_date('centroid', 1)

    get_error_scatter_plots('eccentricity', 'size')
    get_error_scatter_plots('eccentricity', 'mean_intensity')
    get_error_scatter_plots('eccentricity', 'orientation')
    get_error_scatter_plots('mean_intensity', 'size')
    get_error_scatter_plots('mean_intensity', 'orientation')
    get_error_scatter_plots('major_axis_length', 'minor_axis_length')
    get_error_scatter_plots(
        ('centroid', 0, 'X'), ('centroid', 1, 'Y')
    )


if __name__ == '__main__':
    if not os.path.exists('pore_new.db'):
        Base.metadata.create_all(engine)
    base_path = Path(sys.argv[1])
    write_path = Path(sys.argv[2])
    dividor = int(sys.argv[3])
    remainder = int(sys.argv[4])
    seg.do_all(base_path, write_path, dividor, remainder)
    # seg.populate_qs_mesn_and_std(base_path)
