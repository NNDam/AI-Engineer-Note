## Build custom plugins

```
cd nvdsinfer_custom_impl_Yolo
mkdir build && cd build
cmake ..
make -j8
```

## Run deepstream-python
```
LD_PRELOAD=<path-to-plugin.so> python3 run_scrfd.py file:/<path-to-input-video>
```
