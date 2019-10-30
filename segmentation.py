import os
import sys
import datetime
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


normal_field_list = [
    'eccentricity', 'size', 'mean_intensity',
    'major_axis_length', 'minor_axis_length',
    'orientation'
]

comp_field_list = ['weighted_centroid']


k_list = [
    -3.11,
    -3.08,
    -3.14,
    -3.17,
    -3.20,
    -3.26
]


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

        weighted_centroid_list = list()

        for poredata in poredata_list:
            eccentricity_list.append(poredata.eccentricity)
            size_list.append(poredata.size)
            mean_intensity_list.append(poredata.mean_intensity)
            min_intensity_list.append(poredata.min_intensity)
            major_axis_length_list.append(poredata.major_axis_length)
            minor_axis_length_list.append(poredata.minor_axis_length)
            orientation_list.append(poredata.orientation)
            weighted_centroid_list.append(
                list(
                    make_tuple(poredata.weighted_centroid)
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

        a_record.mean_weighted_centroid = str(
            tuple(
                np.mean(
                    weighted_centroid_list, axis=0
                )
            )
        )
        a_record.std_weighted_centroid = str(
            tuple(
                np.std(
                    weighted_centroid_list, axis=0
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

        else:
            x_list.append(
                getattr(a_record, 'mean_' + field2)
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
        else:
            y_list.append(
                getattr(a_record, 'mean_' + field1)
            )

        time_in_sec.append(
            int(
                sunpy.time.parse_time(
                    a_record.date_time.isoformat()
                ).jd * 86400
            )
        )

    time_in_sec -= np.min(time_in_sec)

    time_in_sec = time_in_sec / np.max(time_in_sec)

    time_in_sec *= 256

    time_in_sec = np.int64(time_in_sec)

    colormap = cm.rainbow(
        np.arange(
            np.max(time_in_sec) + 1
        )
    )

    plt.scatter(
        x_list,
        y_list,
        c=colormap[time_in_sec]
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
    if isinstance(field1, tuple):
        plt.savefig(
            '{}_{}_vs_{}_scatter.png'.format(field1[0], field1[2], field2[2]),
            format='png',
            dpi=300
        )
    else:
        plt.savefig(
            '{}_vs_{}_scatter.png'.format(field1, field2),
            format='png',
            dpi=300
        )
    plt.clf()
    plt.cla()


def get_scatter_plots(field1, field2, k_value=None):
    all_records = model.Record.get_all()

    date_list = list()

    k1_dict = defaultdict(list)

    k2_dict = defaultdict(list)

    time_in_sec = list()

    for a_record in all_records:
        date_list.append(a_record.date_time)

        poredata_list = a_record.poredata

        for poredata in poredata_list:
            if isinstance(field2, tuple):
                k1_dict[poredata.k].append(
                    list(make_tuple(getattr(poredata, field2[0])))[field2[1]]
                )

            else:
                k1_dict[poredata.k].append(
                    getattr(poredata, field2)
                )

            if isinstance(field1, tuple):
                k2_dict[poredata.k].append(
                    list(make_tuple(getattr(poredata, field1[0])))[field1[1]]
                )

            else:
                k2_dict[poredata.k].append(
                    getattr(poredata, field1)
                )

        time_in_sec.append(
            int(
                sunpy.time.parse_time(
                    a_record.date_time.isoformat()
                ).jd * 86400
            )
        )

    time_in_sec -= np.min(time_in_sec)

    time_in_sec = time_in_sec / np.max(time_in_sec)

    time_in_sec *= 256

    time_in_sec = np.int64(time_in_sec)

    colormap = cm.rainbow(
        np.arange(
            np.max(time_in_sec) + 1
        )
    )

    plt.scatter(
        k1_dict[k_value],
        k2_dict[k_value],
        c=colormap[time_in_sec]
    )

    if isinstance(field1, tuple):
        plt.title(
            '{} {} vs {} Scatter Plot'.format(
                field1[0], field1[2], field2[2]
            )
        )
        plt.xlabel('{}'.format(field2[2]))
        plt.ylabel('{}'.format(field1[2]))
    else:
        plt.title('{} vs {} Scatter Plot'.format(field1, field2))
        plt.xlabel('{}'.format(field2))
        plt.ylabel('{}'.format(field1))
    plt.legend()
    plt.show()


def error_plot_field_vs_date(field, x_y=0):
    all_records = model.Record.get_all()

    date_list = list()

    value_list = list()

    yerr = list()

    for a_record in all_records:

        date_list.append(a_record.date_time)

        if field in ['weighted_centroid', 'centroid']:
            value = make_tuple(getattr(a_record, 'mean_' + field))[x_y]
            value_list.append(
                value
            )
            valueerr = make_tuple(getattr(a_record, 'std_' + field))[x_y]
            yerr.append(
                valueerr
            )
        else:
            value_list.append(
                getattr(a_record, 'mean_' + field)
            )
            yerr.append(
                getattr(a_record, 'std_' + field)
            )

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    _start_date = datetime.datetime(2017, 9, 28, 8, 51, 20)
    x_ticks = list()
    for i in range(0, 9):
        x_ticks.append(
            _start_date + timedelta(minutes=8 * i)
        )

    plt.errorbar(date_list, value_list, yerr=yerr, fmt='b', ecolor='yellow')

    if field not in ['weighted_centroid', 'centroid']:
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
    if field not in ['weighted_centroid', 'centroid']:
        plt.savefig('{}_vs_time.png'.format(field), format='png', dpi=300)
    else:
        plt.savefig(
            '{}_{}_vs_time.png'.format(field, coord), format='png', dpi=300
        )
    plt.clf()
    plt.cla()


def plot_field_vs_date(field, k_value=None, x_y=0):
    all_records = model.Record.get_all()

    date_list = list()

    k_dict = defaultdict(list)

    # time_in_sec = list()

    for a_record in all_records:
        date_list.append(a_record.date_time)

        poredata_list = a_record.poredata

        for poredata in poredata_list:
            if field in ['weighted_centroid', 'centroid']:
                value = make_tuple(getattr(poredata, field))[x_y]
                k_dict[poredata.k].append(
                    value
                )
            else:
                k_dict[poredata.k].append(
                    getattr(poredata, field)
                )

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    _start_date = datetime.datetime(2017, 9, 28, 8, 51, 20)
    x_ticks = list()
    for i in range(0, 9):
        x_ticks.append(
            _start_date + timedelta(minutes=8 * i)
        )
    for k, value_list in k_dict.items():
        if not k_value or k_value == k:
            plt.plot(date_list, value_list, label='Plot k={}'.format(k))
            plt.scatter(
                date_list, value_list, label='Scatter k={}'.format(k)
            )
    if field not in ['weighted_centroid', 'centroid']:
        plt.title('{} vs Date Time Plot'.format(field))
    else:
        coord = 'x' if x_y == 0 else 'y'
        plt.title('{} {} coordinate vs Date Time Plot'.format(field, coord))
    plt.xlabel('Time')
    plt.ylabel(field)
    plt.xticks(x_ticks, rotation=45)
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.show()


def save_all_plots():
    normal_field_list = [
        'eccentricity', 'size', 'mean_intensity',
        'major_axis_length', 'minor_axis_length',
        'orientation'
    ]

    for a_field in normal_field_list:
        error_plot_field_vs_date(a_field)

    error_plot_field_vs_date('weighted_centroid', 0)
    error_plot_field_vs_date('weighted_centroid', 1)

    get_error_scatter_plots('eccentricity', 'size')
    get_error_scatter_plots('eccentricity', 'mean_intensity')
    get_error_scatter_plots('eccentricity', 'orientation')
    get_error_scatter_plots('mean_intensity', 'size')
    get_error_scatter_plots('mean_intensity', 'orientation')
    get_error_scatter_plots('major_axis_length', 'minor_axis_length')
    get_error_scatter_plots(
        ('weighted_centroid', 0, 'X'), ('weighted_centroid', 1, 'Y')
    )


if __name__ == '__main__':
    if not os.path.exists('pore_new.db'):
        Base.metadata.create_all(engine)
    base_path = Path(sys.argv[1])
    write_path = Path(sys.argv[2])
    dividor = int(sys.argv[3])
    remainder = int(sys.argv[4])
    seg.do_all(base_path, write_path, dividor, remainder)
