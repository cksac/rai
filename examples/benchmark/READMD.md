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
Epoch 0000: train loss:    2.51643, test acc: 11.36%, time: 145.8250297s
Epoch 0001: train loss:    2.29856, test acc: 11.35%, time: 143.4138337s
Epoch 0002: train loss:    2.30091, test acc: 11.35%, time: 143.1113737s
Epoch 0003: train loss:    2.29983, test acc: 11.35%, time: 140.7053021s
Epoch 0004: train loss:    2.29918, test acc: 11.35%, time: 141.6018662s
Epoch 0005: train loss:    2.30104, test acc: 11.35%, time: 142.414461s
Epoch 0006: train loss:    2.30093, test acc: 11.35%, time: 141.254421s
Epoch 0007: train loss:    2.30135, test acc: 11.35%, time: 144.2250752s
Epoch 0008: train loss:    2.30132, test acc: 11.35%, time: 141.0075665s
Epoch 0009: train loss:    2.30088, test acc: 11.35%, time: 140.7315586s
elapsed: 1425.5438951s, avg: 142.55 sec/epoch
```

### Candle
- `cargo run --release -- candle`
```sh
candle mnist training...
device: Cpu
Epoch 0000: train loss:    0.58096, test acc: 91.07%, time: 171.258307s
Epoch 0001: train loss:    0.25766, test acc: 95.38%, time: 175.68223s
Epoch 0002: train loss:    0.18908, test acc: 96.42%, time: 175.6548027s
Epoch 0003: train loss:    0.15013, test acc: 96.15%, time: 168.1546719s
Epoch 0004: train loss:    0.13635, test acc: 97.42%, time: 167.0019699s
Epoch 0005: train loss:    0.12522, test acc: 97.31%, time: 168.6385511s
Epoch 0006: train loss:    0.12127, test acc: 97.78%, time: 167.6541415s
Epoch 0007: train loss:    0.11817, test acc: 97.57%, time: 168.0883122s
Epoch 0008: train loss:    0.12239, test acc: 97.72%, time: 168.4914884s
Epoch 0009: train loss:    0.13028, test acc: 97.49%, time: 166.481768s
elapsed: 1699.1594236s, avg: 169.92 sec/epoch
```

## CUDA
### RAI
- `cargo run --release --features=cuda -- rai`
```sh
rai mnist training...
device: Cuda(0)
Epoch 0000: train loss:    0.47585, test acc: 95.07%, time: 9.1410106s
Epoch 0001: train loss:    0.20546, test acc: 96.36%, time: 12.2727241s
Epoch 0002: train loss:    0.16250, test acc: 97.14%, time: 12.9128751s
Epoch 0003: train loss:    0.13771, test acc: 97.04%, time: 12.2368281s
Epoch 0004: train loss:    0.11819, test acc: 97.28%, time: 13.5339264s
Epoch 0005: train loss:    0.11479, test acc: 97.55%, time: 14.3196856s
Epoch 0006: train loss:    0.10197, test acc: 97.72%, time: 13.0397213s
Epoch 0007: train loss:    0.09014, test acc: 97.90%, time: 13.8322886s
Epoch 0008: train loss:    0.08879, test acc: 98.01%, time: 14.5161161s
Epoch 0009: train loss:    0.08374, test acc: 98.13%, time: 13.7856084s
elapsed: 130.8538784s, avg: 13.09 sec/epoch
```

### Candle
- `cargo run --release --features=cuda -- candle`
```sh
candle mnist training...
device: Cuda(CudaDevice(DeviceId(1)))
Epoch 0000: train loss:    0.44832, test acc: 95.33%, time: 12.1363745s
Epoch 0001: train loss:    0.20236, test acc: 96.60%, time: 11.4870728s
Epoch 0002: train loss:    0.15212, test acc: 97.23%, time: 12.8028247s
Epoch 0003: train loss:    0.13200, test acc: 97.29%, time: 13.6402537s
Epoch 0004: train loss:    0.11891, test acc: 97.62%, time: 15.0222256s
Epoch 0005: train loss:    0.11002, test acc: 97.56%, time: 17.0893168s
Epoch 0006: train loss:    0.10920, test acc: 97.21%, time: 12.5835751s
Epoch 0007: train loss:    0.10493, test acc: 97.76%, time: 12.0987492s
Epoch 0008: train loss:    0.10145, test acc: 97.74%, time: 12.1199194s
Epoch 0009: train loss:    0.09464, test acc: 97.74%, time: 11.5131115s
elapsed: 133.0483271s, avg: 13.30 sec/epoch
```
