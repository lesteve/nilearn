# -*- coding: utf-8 -*-
"""

"""
import os
import shutil
import collections

import pandas as pd

import nibabel

from nilearn import datasets
from nilearn.plotting import plot_stat_map
from nilearn.decomposition.canica import CanICA

from externals.formlayout.formlayout import fedit

from externals import tempita

import report_api as api


def chose_params():
    datalist = [('n_components', 20),
                ('smoothing_fwhm', 6.),
                ('threshold', 3.),
                ('verbose', ['10', '0', '10']),
                ]

    adict = collections.OrderedDict(datalist)

    result = fedit(datalist, title="CanICA",
                   comment="Enter the CanICA parameters")

    params = dict(zip(adict.keys(), result))
    params['verbose'] = int(params['verbose'])
    print 'params: ', params
    return params


def get_fitted_canica(func_files, **params):
    canica = CanICA(memory='nilearn_cache', memory_level=5, random_state=0,
                    n_jobs=-1, **params)

    canica.fit(func_files)
    return canica


def generate_images(components_img):
    # Remove existing images
    images_dir = './report/images'

    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
    os.makedirs(images_dir)
    output_filenames = ['./report/images/IC_{}.png'.format(i)
                        for i in range(params['n_components'])]

    for i, output_file in enumerate(output_filenames):
        plot_stat_map(nibabel.Nifti1Image(components_img.get_data()[..., i],
                                          components_img.get_affine()),
                      display_mode="z", title="IC %d" % i, cut_coords=7,
                      colorbar=False, output_file=output_file)

    # img src in the html needs to be relative to index.html
    img_src_filenames = [os.path.relpath(fn, './report') for fn in output_filenames]

    return img_src_filenames


def generate_report(params_dict, img_src_filenames):
    report = api.Report()

    report.add(api.Section('Model parameters')).add(
        api.Table(params_dict.iteritems(), headers=('Parameter', 'Value'))
    )

    for counter, filename in enumerate(img_src_filenames):
        caption = 'Some caption for component {} goes here'.format(counter)
        section = api.Section('Component #{}'.format(counter)
        ).add(
            api.Paragraph('This is some paragraph text related to '
                          'component #{}'.format(counter))
        ).add(
            api.Image(filename, caption=caption)
        )
        report.add(section)

    return report


if __name__ == '__main__':
    dataset = datasets.fetch_adhd()
    func_files = dataset.func

    params = chose_params()

    canica = get_fitted_canica(func_files, **params)
    # Retrieve the independent components in brain space
    components_img = canica.masker_.inverse_transform(canica.components_)
    img_src_filenames = generate_images(components_img)

    report = generate_report(params, img_src_filenames)
    report.save_html('./report/index.html')
