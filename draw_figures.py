import matplotlib.pyplot as plt


model_x_data = [0.8913, 0.7121, 0.5254, 0.4302, 0.3318]
uniform_x_data = [0.8919, 0.7122, 0.5262, 0.4274, 0.33]
fix_data = [0.8920, 0.7122, 0.5264, 0.4275, 0.3326]
model_y_train_data = [0.9819, 0.9698, 0.8589, 0.6727, 0.4425]
model_y_test_data = [0.9820, 0.9707, 0.8551, 0.6685, 0.4392]
uniform_y_train_data = [0.9845, 0.9822, 0.9762, 0.9711, 0.9647]
uniform_y_test_data = [0.9835, 0.9814, 0.9765, 0.9737, 0.9651]
fix_y_train_data = [0.9855, 0.9704, 0.7308, 0.1854, 0.0349]
fix_y_test_data = [0.9837, 0.9695, 0.7308, 0.1887, 0.0397]
ld = 1
ms = 10
figure = plt.figure()
plt.gca().invert_xaxis()
plt.plot(uniform_x_data, uniform_y_train_data, 'b+--', label="uniform noise/train", linewidth=ld, markersize=ms)
plt.plot(uniform_x_data, uniform_y_test_data, 'bx-', label="uniform noise/test", linewidth=ld, markersize=ms)
plt.plot(fix_data, fix_y_train_data, 'r+--', label="fix noise/train", linewidth=ld, markersize=ms)
plt.plot(fix_data, fix_y_test_data, 'rx-', label="fix noise/test", linewidth=ld, markersize=ms)
plt.plot(model_x_data, model_y_train_data, 'g+--', label="model noise/train", linewidth=ld, markersize=ms)
plt.plot(model_x_data, model_y_test_data, 'gx-', label="model noise/test", linewidth=ld, markersize=ms)
plt.legend()
# plt.xticks([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
plt.xlabel("correct label ratio")
plt.ylabel("classification accuracy")
plt.savefig("accuracy.png")
