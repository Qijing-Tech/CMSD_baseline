# CMSD_baseline
The Baseline Method experiments for Framework of Chinese Medical Synonyms Discovery

[TOC]

### Kmeans

**Tencent Embeddinng**

| DataSet |  \|   | MSKB  |    \|     |  \|  | MDKB  |  \|   |
| :-----: | :---: | :---: | :--: | :--: | :---: | :---: |
| Lab Id  |  ARI  |  NMI  |  FMI  |    ARI  |  NMI  |FMI|
|    1    | 9.07  | 60.87 | 20.23 | 44.44 | 84.76 | 47.13 |
|    2    | 7.76  | 57.66 | 19.08 | 46.47 | 85.61 | 49.10 |
|    3    | 11.36 | 60.7  | 21.37 | 45.03 | 85.11 | 47.68 |
|    4    | 14.28 | 61.79 | 24.28 | 45.65 | 85.09 | 48.09 |
|    5    | 13.24 | 61.95 | 22.82 | 43.42 | 84.26 | 46.05 |

| DataSet |  \|   | smHitSyn |  \|   |  \|   | exHitSyn |  \|   |
| :-----: | :---: | :------: | :---: | :---: | :------: | :---: |
| Lab Id  |  ARI  |   FMI    |  NMI  |  ARI  |   FMI    |  NMI  |
|    1    | 23.25 |  27.14   | 71.23 | 18.91 |  21.58   | 72.77 |
|    2    | 26.61 |  29.54   | 72.61 | 15.41 |  19.07   | 70.66 |
|    3    | 24.25 |  27.70   | 72.01 | 15.36 |  18.98   | 71.12 |
|    4    | 24.70 |  28.06   | 71.22 | 15.69 |  19.50   | 71.54 |
|    5    | 26.13 |  28.82   | 71.73 | 15.74 |  18.90   | 70.92 |

| value (+- std) | ARI         | FMI         | NMI         |
| -------------- | ----------- | ----------- | ----------- |
| MSKB           | 13.78(0.91) | 23.23(0.60) | 62.98(1.52) |
| MDKB           | 45.00(1.04) | 47.61(1.01) | 84.97(0.44) |
| smHitSyn       | 24.99(1.23) | 28.25(0.84) | 71.76(0.52) |
| exHitSyn       | 16.22(1.35) | 19.61(1.01) | 71.40(0.74) |



**BERT Embeddinng**

| DataSet |  \|   | MSKB  |  \|   |  \|   | MDKB  |  \|   |
| :-----: | :---: | :---: | :---: | :---: | :---: | :---: |
| Lab Id  |  ARI  |  FMI  |  NMI  |  ARI  |  FMI  |  NMI  |
|    1    | 48.87 | 50.27 | 80.99 | 64.70 | 65.06 | 90.60 |
|    2    |   44.41   |   45.84   |   80.06   |   64.97   |   65.31   |   90.47   |
|    3    |   49.55   |   50.84   |   83.06   |   66.80   |   67.07   |   90.72   |
|    4    |   46.69   |   48.04   |  80.97  |   64.74   |   65.20   |  90.77  |
|    5    |   46.6   |   48.25   |   80.17   |   65.49   |  65.81  |   90.44   |

| DataSet |  \|   | smHitSyn |  \|   |  \|   | exHitSyn |  \|   |
| :-----: | :---: | :------: | :---: | :---: | :------: | :---: |
| Lab Id  |  ARI  |   FMI    |  NMI  |  ARI  |   FMI    |  NMI  |
|    1    | 30.33 |  32.18   | 74.87 | 29.29 |  30.17   | 76.49 |
|    2    | 32.22 |  33.70   | 74.18 | 26.47 |  27.59   | 75.88 |
|    3    | 28.63 |  30.49   | 73.97 | 25.82 |  26.79   | 75.51 |
|    4    | 31.04 |  33.03   | 75.81 | 26.98 |  28.30   | 76.02 |
|    5    | 31.44 |  32.83   | 73.44 | 28.11 |  29.08   | 75.92 |

| value (+- std) | ARI         | FMI         | NMI         |
| -------------- | ----------- | ----------- | ----------- |
| MSKB           | 47.22(1.83) | 48.65(1.78) | 81.16(1.00) |
| MDKB           |      65.34(0.78)      |     65.69(0.74)     |     90.6 (0.13)     |
| smHitSyn       | 30.73(1.22) | 32.45(1.09) | 74.45(0.82) |
| exHitSyn       | 27.33(1.23) | 28.39(1.17) | 75.96(0.31) |



---



### DBSCAN

**Tencent Embeddinng**

| DataSet |  \|  | MSKB |  \|  |  \|  | MDKB |  \|  |
| :-----: | :--: | :--: | :--: | :--: | :--: | :--: |
| Lab Id  | ARI  | FMI  | NMI  | ARI  | FMI  | NMI  |
|    1    |      |      |      |      |      |      |
|    2    |      |      |      |      |      |      |
|    3    |      |      |      |      |      |      |
|    4    |      |      |      |      |      |      |
|    5    |      |      |      |      |      |      |

| DataSet |  \|  | smHitSyn |  \|  |  \|  | exHitSyn |  \|  |
| :-----: | :--: | :------: | :--: | :--: | :------: | :--: |
| Lab Id  | ARI  |   FMI    | NMI  | ARI  |   FMI    | NMI  |
|    1    | 8.5  |   10.1   | 78.2 | 4.1  |   5.0    | 79.6 |
|    2    |      |          |      |      |          |      |
|    3    |      |          |      |      |          |      |
|    4    |      |          |      |      |          |      |
|    5    |      |          |      |      |          |      |

| value (+- std) | ARI         | FMI         | NMI         |
| -------------- | ----------- | ----------- | ----------- |
| MSKB           | 45.5 (2.33) | 45.5 (2.33) | 45.5 (2.33) |
| MDKB           |             |             |             |
| smHitSyn       | 8.5         | 10.1        | 78.2        |
| exHitSyn       | 4.1         | 5.0         | 79.6        |



**BERT Embeddinng**

| DataSet |  \|   | MSKB  |  \|   |  \|   | MDKB  |  \|   |
| :-----: | :---: | :---: | :---: | :---: | :---: | :---: |
| Lab Id  |  ARI  |  FMI  |  NMI  |  ARI  |  FMI  |  NMI  |
|    1    | 99.99 | 99.99 | 99.99 | 99.99 | 99.99 | 99.99 |
|    2    |       |       |       |       |       |       |
|    3    |       |       |       |       |       |       |
|    4    |       |       |       |       |       |       |
|    5    |       |       |       |       |       |       |

| DataSet |  \|  | smHitSyn |  \|  |  \|  | exHitSyn |  \|  |
| :-----: | :--: | :------: | :--: | :--: | :------: | :--: |
| Lab Id  | ARI  |   FMI    | NMI  | ARI  |   FMI    | NMI  |
|    1    | 3.2  |   7.3    | 79.2 | 0.2  |   1.5    | 82.1 |
|    2    |      |          |      |      |          |      |
|    3    |      |          |      |      |          |      |
|    4    |      |          |      |      |          |      |
|    5    |      |          |      |      |          |      |

| value (+- std) | ARI         | FMI         | NMI         |
| -------------- | ----------- | ----------- | ----------- |
| MSKB           | 45.5 (2.33) | 45.5 (2.33) | 45.5 (2.33) |
| MDKB           |             |             |             |
| smHitSyn       | 3.2         | 7.3         | 79.2        |
| exHitSyn       | 0.2         | 1.5         | 82.1        |



---



### Louvain

**Tencent Embeddinng**

| DataSet |  \|   | MSKB  |  \|   |  \|   | MDKB  |  \|   |
| :-----: | :---: | :---: | :---: | :---: | :---: | :---: |
| Lab Id  |  ARI  |  FMI  |  NMI  |  ARI  |  FMI  |  NMI  |
|    1    | 99.99 | 99.99 | 99.99 | 99.99 | 99.99 | 99.99 |
|    2    |       |       |       |       |       |       |
|    3    |       |       |       |       |       |       |
|    4    |       |       |       |       |       |       |
|    5    |       |       |       |       |       |       |

| DataSet |  \|   | smHitSyn |  \|   |  \|   | exHitSyn |  \|   |
| :-----: | :---: | :------: | :---: | :---: | :------: | :---: |
| Lab Id  |  ARI  |   FMI    |  NMI  |  ARI  |   FMI    |  NMI  |
|    1    | 99.99 |  99.99   | 99.99 | 99.99 |  99.99   | 99.99 |
|    2    |       |          |       |       |          |       |
|    3    |       |          |       |       |          |       |
|    4    |       |          |       |       |          |       |
|    5    |       |          |       |       |          |       |

| value (+- std) | ARI         | FMI         | NMI         |
| -------------- | ----------- | ----------- | ----------- |
| MSKB           | 45.5 (2.33) | 45.5 (2.33) | 45.5 (2.33) |
| MDKB           |             |             |             |
| smHitSyn       |             |             |             |
| exHitSyn       |             |             |             |



**BERT Embeddinng**

| DataSet |  \|   | MSKB  |  \|   |  \|   | MDKB  |  \|   |
| :-----: | :---: | :---: | :---: | :---: | :---: | :---: |
| Lab Id  |  ARI  |  FMI  |  NMI  |  ARI  |  FMI  |  NMI  |
|    1    | 99.99 | 99.99 | 99.99 | 99.99 | 99.99 | 99.99 |
|    2    |       |       |       |       |       |       |
|    3    |       |       |       |       |       |       |
|    4    |       |       |       |       |       |       |
|    5    |       |       |       |       |       |       |

| DataSet |  \|   | smHitSyn |  \|   |  \|   | exHitSyn |  \|   |
| :-----: | :---: | :------: | :---: | :---: | :------: | :---: |
| Lab Id  |  ARI  |   FMI    |  NMI  |  ARI  |   FMI    |  NMI  |
|    1    | 99.99 |  99.99   | 99.99 | 99.99 |  99.99   | 99.99 |
|    2    |       |          |       |       |          |       |
|    3    |       |          |       |       |          |       |
|    4    |       |          |       |       |          |       |
|    5    |       |          |       |       |          |       |

| value (+- std) | ARI         | FMI         | NMI         |
| -------------- | ----------- | ----------- | ----------- |
| MSKB           | 45.5 (2.33) | 45.5 (2.33) | 45.5 (2.33) |
| MDKB           |             |             |             |
| smHitSyn       |             |             |             |
| exHitSyn       |             |             |             |



---



### OPTICS

**Tencent Embeddinng**

| DataSet |  \|   | MSKB  |  \|   |  \|   | MDKB  |  \|   |
| :-----: | :---: | :---: | :---: | :---: | :---: | :---: |
| Lab Id  |  ARI  |  FMI  |  NMI  |  ARI  |  FMI  |  NMI  |
|    1    | 99.99 | 99.99 | 99.99 | 99.99 | 99.99 | 99.99 |
|    2    |       |       |       |       |       |       |
|    3    |       |       |       |       |       |       |
|    4    |       |       |       |       |       |       |
|    5    |       |       |       |       |       |       |

| DataSet |  \|  | smHitSyn |  \|  |  \|  | exHitSyn |  \|  |
| :-----: | :--: | :------: | :--: | :--: | :------: | :--: |
| Lab Id  | ARI  |   FMI    | NMI  | ARI  |   FMI    | NMI  |
|    1    | 1.3  |   10.4   | 52.6 | 0.5  |   7.8    | 48.2 |
|    2    |      |          |      |      |          |      |
|    3    |      |          |      |      |          |      |
|    4    |      |          |      |      |          |      |
|    5    |      |          |      |      |          |      |

| value (+- std) | ARI         | FMI         | NMI         |
| -------------- | ----------- | ----------- | ----------- |
| MSKB           | 45.5 (2.33) | 45.5 (2.33) | 45.5 (2.33) |
| MDKB           |             |             |             |
| smHitSyn       | 1.3         | 10.4        | 52.6        |
| exHitSyn       | 0.5         | 7.8         | 48.2        |



**BERT Embeddinng**

| DataSet |  \|   | MSKB  |  \|   |  \|   | MDKB  |  \|   |
| :-----: | :---: | :---: | :---: | :---: | :---: | :---: |
| Lab Id  |  ARI  |  FMI  |  NMI  |  ARI  |  FMI  |  NMI  |
|    1    | 99.99 | 99.99 | 99.99 | 99.99 | 99.99 | 99.99 |
|    2    |       |       |       |       |       |       |
|    3    |       |       |       |       |       |       |
|    4    |       |       |       |       |       |       |
|    5    |       |       |       |       |       |       |

| DataSet |  \|  | smHitSyn |  \|  |  \|  | exHitSyn |  \|  |
| :-----: | :--: | :------: | :--: | :--: | :------: | :--: |
| Lab Id  | ARI  |   FMI    | NMI  | ARI  |   FMI    | NMI  |
|    1    | 2.7  |   11.1   | 61.8 | 0.9  |   8.0    | 56.7 |
|    2    |      |          |      |      |          |      |
|    3    |      |          |      |      |          |      |
|    4    |      |          |      |      |          |      |
|    5    |      |          |      |      |          |      |

| value (+- std) | ARI         | FMI         | NMI         |
| -------------- | ----------- | ----------- | ----------- |
| MSKB           | 45.5 (2.33) | 45.5 (2.33) | 45.5 (2.33) |
| MDKB           |             |             |             |
| smHitSyn       | 2.7         | 11.1        | 61.8        |
| exHitSyn       | 0.9         | 8.0         | 56.7        |



---



### GMMS

**Tencent Embeddinng**

| DataSet |  \|   | MSKB  |  \|   |  \|   | MDKB  |  \|   |
| :-----: | :---: | :---: | :---: | :---: | :---: | :---: |
| Lab Id  |  ARI  |  FMI  |  NMI  |  ARI  |  FMI  |  NMI  |
|    1    | 14.28 | 22.57 | 63.24 | 47.65 | 49.93 | 85.98 |
|    2    | 12.2  | 22.63 | 60.66 | 43.05 | 45.92 | 84.39 |
|    3    | 13.34 | 23.21 | 62.93 | 42.93 | 45.73 | 83.82 |
|    4    | 14.42 | 24.13 | 65.44 | 40.44 | 43.51 | 83.75 |
|    5    | 14.65 | 23.63 | 62.65 | 43.07 | 46.22 | 84.91 |

| DataSet |  \|   | smHitSyn |  \|   |  \|   | exHitSyn |  \|   |
| :-----: | :---: | :------: | :---: | :---: | :------: | :---: |
| Lab Id  |  ARI  |   FMI    |  NMI  |  ARI  |   FMI    |  NMI  |
|    1    | 21.21 |  25.35   | 71.47 | 14.26 |  17.37   | 69.62 |
|    2    | 20.51 |  25.05   | 69.72 | 14.63 |  18.60   | 70.72 |
|    3    | 22.81 |  27.03   | 72.02 | 14.94 |  18.82   | 70.13 |
|    4    | 21.94 |  26.25   | 70.43 | 17.36 |  20.29   | 71.56 |
|    5    | 26.55 |  29.80   | 72.69 | 17.67 |  20.31   | 71.07 |

| value (+- std) | ARI         | FMI         | NMI         |
| -------------- | ----------- | ----------- | ----------- |
| MSKB           | 13.78(0.91) | 23.23(0.60) | 62.98(1.52) |
| MDKB           | 43.43(2.34) | 46.26(2.07) | 84.57(0.82) |
| smHitSyn       | 22.60(2.11) | 26.70(1.70) | 71.27(1.07) |
| exHitSyn       | 15.77(1.44) | 19.08(1.11) | 70.62(0.69) |



---



**BERT Embeddinng**

| DataSet |  \|   | MSKB  |  \|   |  \|   | MDKB  |  \|   |
| :-----: | :---: | :---: | :---: | :---: | :---: | :---: |
| Lab Id  |  ARI  |  FMI  |  NMI  |  ARI  |  FMI  |  NMI  |
|    1    | 43.49 | 45.02 | 79.84 | 66.45 | 66.80 | 91.22 |
|    2    |  43.92  |   45.92   |  78.50  |  65.59  | 65.98  |   90.51   |
|    3    | 41.94 |   43.47   |   78.92   |    62.12    |   62.44   |  89.42  |
|    4    |  45.39  |   46.98   |   78.59   |    62.15    |   62.61   | 89.77 |
|    5    |    43.55    |   45.33   |   80.91   | 63.29 |  63.29  |   90.07   |

| DataSet |  \|   | smHitSyn |  \|   |  \|   | exHitSyn |  \|   |
| :-----: | :---: | :------: | :---: | :---: | :------: | :---: |
| Lab Id  |  ARI  |   FMI    |  NMI  |  ARI  |   FMI    |  NMI  |
|    1    | 32.64 |  34.00   | 74.64 | 27.79 |  28.89   | 75.80 |
|    2    | 31.41 |  33.15   | 74.54 | 28.51 |  29.48   | 76.07 |
|    3    | 31.32 |  32.68   | 73.79 | 26.95 |  27.97   | 75.39 |
|    4    | 28.57 |  30.22   | 73.00 | 26.83 |  28.34   | 76.30 |
|    5    | 29.25 |  31.36   | 74.28 | 23.58 |  25.14   | 75.40 |

| value (+- std) | ARI         | FMI         | NMI         |
| -------------- | ----------- | ----------- | ----------- |
| MSKB           | 43.49(1.10) | 45.35(1.15) | 79.35(0.91) |
| MDKB           |     63.92(1.79)     |  64.30(11.78)  |          90.20(0.62)          |
| smHitSyn       | 30.73(1.22) | 32.45(1.09) | 74.45(0.82) |
| exHitSyn       | 26.73(1.69) | 27.97(1.50) | 75.79(0.36) |



---



### AC

**Tencent Embeddinng**

| DataSet |  \|   | MSKB  |  \|   |  \|   | MDKB  |  \|   |
| :-----: | :---: | :---: | :---: | :---: | :---: | :---: |
| Lab Id  |  ARI  |  FMI  |  NMI  |  ARI  |  FMI  |  NMI  |
|    1    | 19.44 | 28.33 | 66.89 | 48.71 | 51.28 | 87.43 |
|    2    |       |       |       |       |       |       |
|    3    |       |       |       |       |       |       |
|    4    |       |       |       |       |       |       |
|    5    |       |       |       |       |       |       |

| DataSet |  \|   | smHitSyn |  \|   |  \|   | exHitSyn |  \|   |
| :-----: | :---: | :------: | :---: | :---: | :------: | :---: |
| Lab Id  |  ARI  |   FMI    |  NMI  |  ARI  |   FMI    |  NMI  |
|    1    | 33.21 |  35.05   | 75.95 | 26.40 |  28.17   | 77.11 |
|    2    |       |          |       |       |          |       |
|    3    |       |          |       |       |          |       |
|    4    |       |          |       |       |          |       |
|    5    |       |          |       |       |          |       |

| value (+- std) | ARI   | FMI   | NMI   |
| -------------- | ----- | ----- | ----- |
| MSKB           | 19.44 | 28.33 | 66.89 |
| MDKB           | 48.71 | 51.28 | 87.43 |
| smHitSyn       | 33.21 | 35.05 | 75.95 |
| exHitSyn       | 26.40 | 28.17 | 77.11 |





**BERT Embeddinng**

| DataSet |  \|   | MSKB  |  \|   |  \|   | MDKB  |  \|   |
| :-----: | :---: | :---: | :---: | :---: | :---: | :---: |
| Lab Id  |  ARI  |  FMI  |  NMI  |  ARI  |  FMI  |  NMI  |
|    1    | 49.84 | 51.07 | 82.11 | 69.13 | 69.43 | 92.00 |
|    2    |       |       |       |       |       |       |
|    3    |       |       |       |       |       |       |
|    4    |       |       |       |       |       |       |
|    5    |       |       |       |       |       |       |

| DataSet |  \|  | smHitSyn |  \|  |  \|  | exHitSyn |  \|  |
| :-----: | :--: | :------: | :--: | :--: | :------: | :--: |
| Lab Id  | ARI  |   FMI    | NMI  | ARI  |   FMI    | NMI  |
|    1    | 39.26 |  40.35   | 77.19 | 32.22 |  33.16   | 78.68 |
|    2     |       |          |       |       |          |       |
|    3     |       |          |       |       |          |       |
|    4    |       |          |       |       |          |       |
|    5     |       |          |       |       |          |       |

| value (+- std) | ARI         | FMI         | NMI         |
| -------------- | ----------- | ----------- | ----------- |
| MSKB           | 49.84 | 51.07 | 82.11 |
| MDKB          | 69.13 | 69.43 | 92.00 |
| smHitSyn       | 39.26 | 40.35 | 77.19 |
| exHitSyn       | 32.22 | 33.16 | 78.68 |

---



### L2C

**Tencent Embeddinng**

| DataSet |  \|   | MSKB  |  \|   |  \|   | MDKB  |  \|   |
| :-----: | :---: | :---: | :---: | :---: | :---: | :---: |
| Lab Id  |  ARI  |  FMI  |  NMI  |  ARI  |  FMI  |  NMI  |
|    1    | 99.99 | 99.99 | 99.99 | 99.99 | 99.99 | 99.99 |
|    2    |       |       |       |       |       |       |
|    3    |       |       |       |       |       |       |
|    4    |       |       |       |       |       |       |
|    5    |       |       |       |       |       |       |

| DataSet |  \|   | smHitSyn |  \|   |  \|   | exHitSyn |  \|   |
| :-----: | :---: | :------: | :---: | :---: | :------: | :---: |
| Lab Id  |  ARI  |   FMI    |  NMI  |  ARI  |   FMI    |  NMI  |
|    1    | 99.99 |  99.99   | 99.99 | 99.99 |  99.99   | 99.99 |
|    2    |       |          |       |       |          |       |
|    3    |       |          |       |       |          |       |
|    4    |       |          |       |       |          |       |
|    5    |       |          |       |       |          |       |

| value (+- std) | ARI         | FMI         | NMI         |
| -------------- | ----------- | ----------- | ----------- |
| MSKB           | 45.5 (2.33) | 45.5 (2.33) | 45.5 (2.33) |
| MDKB           |             |             |             |
| smHitSyn       |             |             |             |
| exHitSyn       |             |             |             |



**BERT Embeddinng**

| DataSet |  \|   | MSKB  |  \|   |  \|   | MDKB  |  \|   |
| :-----: | :---: | :---: | :---: | :---: | :---: | :---: |
| Lab Id  |  ARI  |  FMI  |  NMI  |  ARI  |  FMI  |  NMI  |
|    1    | 99.99 | 99.99 | 99.99 | 99.99 | 99.99 | 99.99 |
|    2    |       |       |       |       |       |       |
|    3    |       |       |       |       |       |       |
|    4    |       |       |       |       |       |       |
|    5    |       |       |       |       |       |       |

| DataSet |  \|   | smHitSyn |  \|   |  \|   | exHitSyn |  \|   |
| :-----: | :---: | :------: | :---: | :---: | :------: | :---: |
| Lab Id  |  ARI  |   FMI    |  NMI  |  ARI  |   FMI    |  NMI  |
|    1    | 99.99 |  99.99   | 99.99 | 99.99 |  99.99   | 99.99 |
|    2    |       |          |       |       |          |       |
|    3    |       |          |       |       |          |       |
|    4    |       |          |       |       |          |       |
|    5    |       |          |       |       |          |       |

| value (+- std) | ARI         | FMI         | NMI         |
| -------------- | ----------- | ----------- | ----------- |
| MSKB           | 45.5 (2.33) | 45.5 (2.33) | 45.5 (2.33) |
| MDKB           |             |             |             |
| smHitSyn       |             |             |             |
| exHitSyn       |             |             |             |



