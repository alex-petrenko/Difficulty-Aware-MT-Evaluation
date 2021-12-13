import os
import sys
from os.path import join

import numpy as np
import pandas
import scipy


def read_dataframe(lp, systems, year):
    csv_path = join('res', str(year), lp, 'results.csv')
    df = pandas.read_csv(csv_path)
    systems_to_drop = []
    for i in range(len(df['SYSTEM'])):
        if df['SYSTEM'][i] not in systems:
            systems_to_drop.append(i)
    df.drop(systems_to_drop, axis=0, inplace=True)
    df.sort_values(by=['HUMAN'], ascending=False, inplace=True)
    df = df.reset_index()
    return df


def normalised_kendall_tau_distance(values1, values2):
    """Compute the Kendall tau distance."""
    n = len(values1)
    assert len(values2) == n, "Both lists have to be of equal length"
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = np.argsort(values1)
    b = np.argsort(values2)
    ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
    return ndisordered / (n * (n - 1))


def main():
    years = [2017, 2018, 2019, 2020]
    language_pairs = ['en-de', 'de-en']

    metrics = ['BLEU', 'TER', 'METEOR', 'BERT_F', 'HUG_BERT_F', 'DA_BERT_F', 'HUMAN']
    col_names = ['BLEU ↑', 'TER ↓', 'METEOR ↑', 'BERTScore ↑', 'BERTScoreHF ↑', 'DA-BERTScore ↑', 'HUMAN ↑']

    systems_per_year = {
        2017: {
            'en-de': {
                'C-3MA.4959': 'C-3MA.4959',
                'fbk-nmt-combination.4870': 'fbk-nmt-combination.4870',
                'KIT.4950': 'KIT.4950',
                'LIUM-NMT.4900': 'LIUM-NMT.4900',
                'LMU-nmt-reranked.4934': 'LMU-nmt-reranked.4934',
                'LMU-nmt-single.4893': 'LMU-nmt-single.4893',
                'online-A.0': 'online-A.0',
                'online-B.0': 'online-B.0',
                'online-F.0': 'online-F.0',
                'online-G.0': 'online-G.0',
                'PROMT-Rule-based.4735': 'PROMT-Rule-based.4735',
                'RWTH-nmt-ensemble.4921': 'RWTH-nmt-ensemble.4921',
                'SYSTRAN.4847': 'SYSTRAN.4847',
                'TALP-UPC.4834': 'TALP-UPC.4834',
                'uedin-nmt.4722': 'uedin-nmt.4722',
                'xmu.4910': 'xmu.4910',
            },
            'de-en': {
                'C-3MA.4958': 'C-3MA.4958',
                'KIT.4951': 'KIT.4951',
                'LIUM-NMT.4733': 'LIUM-NMT.4733',
                'online-A.0': 'online-A.0',
                'online-B.0': 'online-B.0',
                'online-F.0': 'online-F.0',
                'online-G.0': 'online-G.0',
                'RWTH-nmt-ensemble.4920': 'RWTH-nmt-ensemble.4920',
                'SYSTRAN.4846': 'SYSTRAN.4846',
                'TALP-UPC.4830': 'TALP-UPC.4830',
                'uedin-nmt.4723': 'uedin-nmt.4723',
            },
        },
        2018: {
            'en-de': {
                'JHU.5703': 'JHU.5703',
                'KIT.5486': 'KIT.5486',
                'LMU-nmt.5741': 'LMU-nmt.5741',
                'LMU-unsup.5651': 'LMU-unsup.5651',
                'Microsoft-Marian.5691': 'Microsoft-Marian.5691',
                'MMT-production-system.5594': 'MMT-production-system.5594',
                'NTT.5667': 'NTT.5667',
                'online-A.0': 'online-A.0',
                'online-B.0': 'online-B.0',
                'online-F.0': 'online-F.0',
                'online-G.0': 'online-G.0',
                'online-Y.0': 'online-Y.0',
                'online-Z.0': 'online-Z.0',
                'RWTH-UNSUPER.5484': 'RWTH-UNSUPER.5484',
                'UCAM.5585': 'UCAM.5585',
                'uedin.5770': 'uedin.5770',
            },
            'de-en': {
                'JHU.5706': 'JHU.5706',
                'LMU-nmt.5756': 'LMU-nmt.5756',
                'LMU-unsup.5650': 'LMU-unsup.5650',
                'MLLP-UPV.5554': 'MLLP-UPV.5554',
                'NJUNMT-private.5406': 'NJUNMT-private.5406',
                'NTT.5666': 'NTT.5666',
                'online-A.0': 'online-A.0',
                'online-B.0': 'online-B.0',
                'online-F.0': 'online-F.0',
                'online-G.0': 'online-G.0',
                'online-Y.0': 'online-Y.0',
                'RWTH.5636': 'RWTH.5636',
                'RWTH-UNSUPER.5482': 'RWTH-UNSUPER.5482',
                'Ubiqus-NMT.5635': 'Ubiqus-NMT.5635',
                'UCAM.5690': 'UCAM.5690',
                'uedin.5766': 'uedin.5766',
            },
        },
        2019: {
            'en-de': {
                'Facebook_FAIR.6862': 'Facebook.6862',
                'Microsoft-WMT19-sentence_document.6974': 'Microsoft.sd.6974',
                'Microsoft-WMT19-document-level.6808': 'Microsoft.dl.6808',
                'MSRA.MADL.6926': 'MSRA.6926',
                'UCAM.6731': 'UCAM.6731',
                'NEU.6763': 'NEU.6763',
                'dfki-nmt.6479': 'dfki-nmt.6479',
                'eTranslation.6823': 'eTranslation.6823',
                'Helsinki-NLP.6820': 'Helsinki-NLP.6820',
                'JHU.6819': 'JHU.6819',
                'lmu-ctx-tf-single-en-de.6981': 'lmu-ctx.6981',
                'MLLP-UPV.6651': 'MLLP-UPV.6651',
                'online-A.0': 'online-A.0',
                'online-B.0': 'online-B.0',
                'online-G.0': 'online-G.0',
                'online-X.0': 'online-X.0',
                'online-Y.0': 'online-Y.0',
                'PROMT_NMT_EN-DE.6674': 'PROMT.6674',
                'TartuNLP-c.6508': 'TartuNLP.6508',
                'UdS-DFKI.6871': 'UdS-DFKI.6871',
            },
            'en-deNOT USED': {
                'Facebook_FAIR.6862': 'Facebook.6862',
                'Microsoft-WMT19-sentence_document.6974': 'Microsoft.sd.6974',
                'Microsoft-WMT19-document-level.6808': 'Microsoft.dl.6808',
                'MSRA.MADL.6926': 'MSRA.6926',
                'UCAM.6731': 'UCAM.6731',
                'NEU.6763': 'NEU.6763',
            },
            'de-en': {
                'dfki-nmt.6478': 'dfki-nmt.6478',
                'Facebook_FAIR.6750': 'Facebook_FAIR.6750',
                'JHU.6809': 'JHU.6809',
                'MLLP-UPV.6899': 'MLLP-UPV.6899',
                'MSRA.MADL.6910': 'MSRA.MADL.6910',
                'NEU.6801': 'NEU.6801',
                'online-A.0': 'online-A.0',
                'online-B.0': 'online-B.0',
                'online-G.0': 'online-G.0',
                'online-Y.0': 'online-Y.0',
                'PROMT_NMT_DE-EN.6683': 'PROMT_NMT_DE-EN.6683',
                'RWTH_Aachen_System.6818': 'RWTH_Aachen_System.6818',
                'TartuNLP-c.6502': 'TartuNLP-c.6502',
                'UCAM.6461': 'UCAM.6461',
                'uedin.6749': 'uedin.6749',
            },
        },
        2020: {
            'en-de': {
                'en-de.AFRL.1069': 'AFRL.1069',
                'en-de.eTranslation.737': 'eTranslation.737',
                'en-de.Huoshan_Translate.832': 'Huoshan_Translate.832',
                'en-de.Online-A.1574': 'Online-A.1574',
                'en-de.Online-B.1590': 'Online-B.1590',
                'en-de.Online-G.1556': 'Online-G.1556',
                'en-de.Online-Z.1631': 'Online-Z.1631',
                'en-de.OPPO.1535': 'OPPO.1535',
                'en-de.PROMT_NMT.73': 'PROMT_NMT.73',
                'en-de.Tencent_Translation.1520': 'Tencent_Translation.1520',
                'en-de.Tohoku-AIP-NTT.890': 'Tohoku-AIP-NTT.890',
                'en-de.UEDIN.1136': 'UEDIN.1136',
                'en-de.WMTBiomedBaseline.388': 'WMTBiomedBaseline.388',
                'en-de.zlabs-nlp.179': 'zlabs-nlp.179',
            },
            'de-en': {
                'de-en.Huoshan_Translate.789': 'Huoshan_Translate.789',
                'de-en.Online-A.1571': 'Online-A.1571',
                'de-en.Online-B.1587': 'Online-B.1587',
                'de-en.Online-G.1553': 'Online-G.1553',
                'de-en.Online-Z.1629': 'Online-Z.1629',
                'de-en.OPPO.1360': 'OPPO.1360',
                'de-en.PROMT_NMT.77': 'PROMT_NMT.77',
                'de-en.Tohoku-AIP-NTT.1442': 'Tohoku-AIP-NTT.1442',
                'de-en.UEDIN.1066': 'UEDIN.1066',
                'de-en.WMTBiomedBaseline.387': 'WMTBiomedBaseline.387',
                'de-en.yolo.1052': 'yolo.1052',
                'de-en.zlabs-nlp.1153': 'zlabs-nlp.1153',
            },
        },
    }

    for year in years:
        for lp in language_pairs:
            systems = systems_per_year[year][lp]

            table = [
                r'\begin{table*}[ht]',
                r'\centering',
                r'\tiny',
                r'\begin{tabular}{' + 'c' * (len(metrics) + 1) + '}',
                r'\toprule',
            ]

            header_tokens = [f'\\textbf{{{c}}}' for c in ['SYSTEM'] + col_names]
            header = ' & '.join(header_tokens) + r' \\'
            table.extend([
                header,
                r'\midrule',
            ])

            table_body = []

            df = read_dataframe(lp, systems, year)

            sum_delta = dict.fromkeys(metrics, 0)

            for row_idx in range(len(df)):
                system = df['SYSTEM'][row_idx]
                system_name = systems[system].replace('_', '-')

                row_tokens = [f'\\textbf{{{system_name}}}']

                human_rank = list(reversed(sorted(df['HUMAN']))).index(df['HUMAN'][row_idx])

                for i, c in enumerate(metrics):
                    higher_value_is_good = col_names[i][-1] == '↑'
                    value = df[c][row_idx]
                    rank = sorted(df[c]).index(value)

                    if higher_value_is_good:
                        rank = len(df[c]) - 1 - rank

                    is_best = rank == 0

                    delta = rank - human_rank
                    chg_direction = ''
                    if c != 'HUMAN':
                        if delta < 0:
                            chg_direction = f' ($\\Uparrow${-delta})'
                        elif delta > 0:
                            chg_direction = f' ($\\Downarrow${delta})'
                        else:
                            chg_direction = f' (\\checkmark{delta})'

                    bold = '\\textbf' if is_best else ''
                    row_tokens.append(f'{bold}{{{value:.4f}{chg_direction}}}')

                    sum_delta[c] += abs(delta)

                row = ' & '.join(row_tokens)
                table_body.append(row)

            table_body = ' \\\\\n'.join(table_body)

            deltas_tokens = []
            for c in metrics:
                value = sum_delta[c]
                is_best = sorted(sum_delta.values()).index(value) == 1
                bold = '\\textbf' if is_best else ''
                token = f'{bold}{{{sum_delta[c]}}}'
                deltas_tokens.append(token)
            deltas = f'$sum(|\\Delta_{{rank}}|)$ & {" & ".join(deltas_tokens)} \\\\'

            table.extend([
                table_body + r' \\',
                r'\midrule',
                deltas,
                r'\bottomrule',
                r'\end{tabular}',
                f'\\caption{{\\label{{tab:wmt_{year}_{lp}}}WMT{year}, {lp} translation.}}',
                r'\end{table*}',
            ])

            table = '\n'.join(table)

            output_dir_name = join('tables', str(year))
            os.makedirs(output_dir_name, exist_ok=True)

            with open(join(output_dir_name, f'{year}_{lp}_table2.tex'), 'w') as fobj:
                fobj.write(table)

            print(table + '\n\n')
            del table

    print('\n\n')

    # now calc Table 1
    num_columns = 1 + len(years) * len(language_pairs)
    table_tau = [
        r'\begin{table*}[ht]',
        r'\centering',
        r'\tiny',
        r'\begin{tabular}{' + 'c' * num_columns + '}',
        r'\toprule',
    ]
    header_tokens = ['\\textbf{METRIC}']
    for year in years:
        for lp in language_pairs:
            lp_tokens = lp.split('-')
            lp_tokens = [t.capitalize() for t in lp_tokens]
            lp_dir = '$\\rightarrow$'.join(lp_tokens)
            header_tokens.append(f'\\textbf{{{year} {lp_dir}}}')

    header = ' & '.join(header_tokens) + r' \\'
    table_tau.extend([
        header,
        r'\midrule',
    ])

    tau_per_competition = dict()

    for year in years:
        for lp in language_pairs:
            systems = systems_per_year[year][lp]
            df = read_dataframe(lp, systems, year)
            corr = df.corr(method='kendall')

            for m_idx, m in enumerate(metrics):
                if m == 'HUMAN':
                    continue

                tau = abs(corr[m]['HUMAN'])

                if (year, lp) not in tau_per_competition:
                    tau_per_competition[(year, lp)] = []
                tau_per_competition[(year, lp)].append(tau)

    table_body = []

    for m_idx, m in enumerate(metrics):
        if m == 'HUMAN':
            continue

        metric_name = col_names[m_idx][:-1].replace('_', '-')
        row_tokens = [f'\\textbf{{{metric_name}}}']

        for year in years:
            for lp in language_pairs:
                values = tau_per_competition[(year, lp)]
                value = values[m_idx]
                is_best = list(reversed(sorted(values))).index(value) == 0
                bold = '\\textbf' if is_best else ''
                token = f'{bold}{{{value:.4f}}}'
                row_tokens.append(token)

        row = ' & '.join(row_tokens)
        table_body.append(row)

    table_body = ' \\\\\n'.join(table_body)

    table_tau.extend([
        table_body + r' \\',
        r'\bottomrule',
        r'\end{tabular}',
        f'\\caption{{\\label{{tab:table1_tau}}Kendall tau distance for different metrics.}}',
        r'\end{table*}',
    ])

    table_tau = '\n'.join(table_tau)

    output_dir_name = join('tables', 'table1_tau')
    os.makedirs(output_dir_name, exist_ok=True)

    with open(join(output_dir_name, f'table1.tex'), 'w') as fobj:
        fobj.write(table_tau)

    print(table_tau + '\n\n')
    del table_tau


if __name__ == '__main__':
    sys.exit(main())
