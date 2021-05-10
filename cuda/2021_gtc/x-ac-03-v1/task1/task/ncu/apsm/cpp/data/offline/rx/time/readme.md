# data folder

This test data is used for offline mode in APSM_gui and APSM_cli.

Test data is converted to synchronized data by using APSM_converter.
This synchronized data is for the APSM_tool, which is a simplified version with less lib dependencies.

## test data

Data format structure for the test data.

| Value  | Format   | Description   | Comment                |
|--------|----------|---------------|------------------------|
| 1      | int32    | Num keys      | total keys             |
| 5      | int32    | Len name 1    | key 1 name length      |
| [...]  | char[]   | [rxSig]       | key 1 name             |
| 3      | int32    | Dimensions    | num dims for key 1     |
| 153600 | int32    | Dim 1         | dim 1 length           |
| 16     | int32    | Dim 2         | dim 2 length           |
| 2      | int32    | Dim 3         | dim 3 length           |
| [...]  | float[]  | Data: 4915200 | 15360 x 16 x 2         |


### rxData_BPSK_alltx.bin

```console
rxSig: [153600, 16, 2]
```

### rxData_QPSK.bin

! only the first 3 users are active, the others are muted.

```console
rxSig: [153600, 16, 2]
```

### rxData_QPSK_alltx.bin

```console
rxSig: [153600, 16, 2]
```

### rxData_QAM16_alltx.bin

```console
rxSig: [153600, 16, 2]
```

### rxData_QAM64_alltx.bin

```console
rxSig: [153600, 16, 2]
```

## synchronized data


| Value | Format   | Description     | Comment                |
|-------|----------|-----------------|------------------------|
| 2     | int32    | Num keys        | total keys             |
| 13    | int32    | Len name 1      | key 1 name length      |
| [...] | char[]   | [rxSigTraining] | key 1 name             |
| 9     | int32    | Len name 2      | key 2 name length      |
| [...] | char[]   | [rxSigData]     | key 2 name             |
| 3     | int32    | Dimensions      | num dims for key 1     |
| 685   | int32    | Dim 1           | dim 1 length           |
| 16    | int32    | Dim 2           | dim 2 length           |
| 2     | int32    | Dim 3           | dim 3 length           |
| [...] | float[]  | Data: 21920     | 685 x 16 x 2           |
| 3     | int32    | Dimensions      | num dims for key 2     |
| 3840  | int32    | Dim 1           | dim 1 length           |
| 16    | int32    | Dim 2           | dim 2 length           |
| 2     | int32    | Dim 3           | dim 3 length           |
| [...] | float[]  | Data: 122880    | 3840 x 16 x 2          |


### rxData_BPSK_alltx_converted.bin

```console
rxSigTraining: [685, 16, 2]
rxSigData: [3840, 16, 2]
```

### rxData_QPSK_alltx_converted.bin

```console
rxSigTraining: [685, 16, 2]
rxSigData: [3840, 16, 2]
```

### rxData_QAM16_alltx_converted.bin

```console
rxSigTraining: [685, 16, 2]
rxSigData: [3840, 16, 2]
```

### rxData_QAM64_alltx_converted.bin

```console
rxSigTraining: [685, 16, 2]
rxSigData: [3840, 16, 2]
```
