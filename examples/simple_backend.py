import xrtoy

# Generate data
frame_shape = (2, 8, 12)
filenames = xrtoy.save_data(
    "/tmp/mydata",
    frame_shape,
    2,
    n_files=2,
    mode="w",
)

print(filenames)
