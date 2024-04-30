# Benchmarks

## Machine
```
CPU: Intel(R) Core(TM) i9-9900X CPU @ 3.50GHz   3.51 GHz
RAM: 128 GB
GPU 0: NVIDIA GeForce RTX 2080 Ti
GPU 1: NVIDIA GeForce RTX 2080 Ti
```

## CPU
### RAI
- `cargo run --release -- rai`
```sh
rai mnist training...
device: Cpu
epoch: 0000, train loss:    2.51643, test acc: 11.36%, time: 145.8250297s
epoch: 0001, train loss:    2.29856, test acc: 11.35%, time: 143.4138337s
epoch: 0002, train loss:    2.30091, test acc: 11.35%, time: 143.1113737s
epoch: 0003, train loss:    2.29983, test acc: 11.35%, time: 140.7053021s
epoch: 0004, train loss:    2.29918, test acc: 11.35%, time: 141.6018662s
epoch: 0005, train loss:    2.30104, test acc: 11.35%, time: 142.414461s
epoch: 0006, train loss:    2.30093, test acc: 11.35%, time: 141.254421s
epoch: 0007, train loss:    2.30135, test acc: 11.35%, time: 144.2250752s
epoch: 0008, train loss:    2.30132, test acc: 11.35%, time: 141.0075665s
epoch: 0009, train loss:    2.30088, test acc: 11.35%, time: 140.7315586s
elapsed: 1425.5438951s, avg: 142.55 sec/epoch
```

### Candle
- `cargo run --release -- candle`
```sh
candle mnist training...
device: Cpu
epoch: 0000, train loss:    0.58096, test acc: 91.07%, time: 171.258307s
epoch: 0001, train loss:    0.25766, test acc: 95.38%, time: 175.68223s
epoch: 0002, train loss:    0.18908, test acc: 96.42%, time: 175.6548027s
epoch: 0003, train loss:    0.15013, test acc: 96.15%, time: 168.1546719s
epoch: 0004, train loss:    0.13635, test acc: 97.42%, time: 167.0019699s
epoch: 0005, train loss:    0.12522, test acc: 97.31%, time: 168.6385511s
epoch: 0006, train loss:    0.12127, test acc: 97.78%, time: 167.6541415s
epoch: 0007, train loss:    0.11817, test acc: 97.57%, time: 168.0883122s
epoch: 0008, train loss:    0.12239, test acc: 97.72%, time: 168.4914884s
epoch: 0009, train loss:    0.13028, test acc: 97.49%, time: 166.481768s
elapsed: 1699.1594236s, avg: 169.92 sec/epoch
```

## CUDA
### RAI
- `cargo run --release --features=cuda -- rai`
```sh
rai mnist training...
device: Cuda(0)
epoch: 0000, train loss:    0.42633, test acc: 95.47%, time: 8.753078s
epoch: 0001, train loss:    0.19833, test acc: 96.97%, time: 7.4540907s
epoch: 0002, train loss:    0.14948, test acc: 97.19%, time: 9.2297638s
epoch: 0003, train loss:    0.12684, test acc: 97.42%, time: 9.0077766s
epoch: 0004, train loss:    0.11815, test acc: 97.80%, time: 8.7115078s
epoch: 0005, train loss:    0.10893, test acc: 97.93%, time: 8.2589094s
epoch: 0006, train loss:    0.09944, test acc: 97.88%, time: 8.6236485s
epoch: 0007, train loss:    0.09280, test acc: 97.78%, time: 9.0383323s
epoch: 0008, train loss:    0.09160, test acc: 97.51%, time: 8.4583471s
epoch: 0009, train loss:    0.08854, test acc: 97.79%, time: 8.4442648s
elapsed: 85.9852372s, avg: 8.60 sec/epoch
```

### Candle
- `cargo run --release --features=cuda -- candle`
```sh
candle mnist training...
device: Cuda(CudaDevice(DeviceId(1)))
epoch: 0000, train loss:    0.47332, test acc: 95.34%, time: 11.3321441s
epoch: 0001, train loss:    0.19311, test acc: 96.85%, time: 10.3816011s
epoch: 0002, train loss:    0.14299, test acc: 97.26%, time: 10.8846721s
epoch: 0003, train loss:    0.12411, test acc: 97.37%, time: 13.5326364s
epoch: 0004, train loss:    0.11417, test acc: 97.60%, time: 16.407464s
epoch: 0005, train loss:    0.10599, test acc: 97.45%, time: 14.7056559s
epoch: 0006, train loss:    0.10664, test acc: 97.48%, time: 12.4014647s
epoch: 0007, train loss:    0.09935, test acc: 97.93%, time: 12.5538107s
epoch: 0008, train loss:    0.09342, test acc: 97.95%, time: 12.0582781s
epoch: 0009, train loss:    0.09422, test acc: 98.23%, time: 12.0370516s
elapsed: 126.3006071s, avg: 12.63 sec/epoch
```
