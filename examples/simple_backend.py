import xrtoy

# Generate data
frame_shape = (80, 244, 324)
filenames = xrtoy.save_data(
    "/tmp/mydata",
    frame_shape,
    2,
    n_files=2,
    mode="w",
)

print(filenames)
