import matplotlib.pyplot as plt


# model_x_data = [0.8913, 0.8020, 0.7121, 0.6229, 0.5254, 0.4302, 0.3318, 0.2333]
# uniform_x_data = [0.8919, 0.7989, 0.7122, 0.6213, 0.5262, 0.4274, 0.33, 0.2354]
# fix_data = [0.8920, 0.8054, 0.7122, 0.6248, 0.5264, 0.4275, 0.3326, 0.2335]
#
# # model_y_train_data = [0.9819, 0.9698, 0.8589, 0.6727, 0.4425]
# model_y_test_data = [0.9820, 0.9756, 0.9707, 0.9401, 0.8551, 0.6685, 0.4392, 0.2237]
# # uniform_y_train_data = [0.9845, 0.9822, 0.9762, 0.9711, 0.9647]
# uniform_y_test_data = [0.9835, 0.9814, 0.9814, 0.9826, 0.9765, 0.9737, 0.9651, 0.9557]
# # fix_y_train_data = [0.9855, 0.9704, 0.7308, 0.1854, 0.0349]
# fix_y_test_data = [0.9837, 0.9795, 0.9695, 0.9098, 0.5707, 0.1887, 0.0397, 0.0066]
# ld = 1
# ms = 10
# figure = plt.figure()
# plt.gca().invert_xaxis()
# # plt.plot(uniform_x_data, uniform_y_train_data, 'b+--', label="uniform noise/train", linewidth=ld, markersize=ms)
# plt.plot(uniform_x_data, uniform_y_test_data, 'bs-', label="Uniform Noise", linewidth=ld, markersize=ms)
# # plt.plot(fix_data, fix_y_train_data, '+--', label="fix noise/train", linewidth=ld, markersize=ms)
# plt.plot(fix_data, fix_y_test_data, 'rv-.', label="Bias Noise", linewidth=ld, markersize=ms)
# # plt.plot(model_x_data, model_y_train_data, 'g+--', label="model noise/train", linewidth=ld, markersize=ms)
# plt.plot(model_x_data, model_y_test_data, 'g^--', label="Generated Noise", linewidth=ld, markersize=ms)
# legend_font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 13}
# plt.legend(prop=legend_font)
# plt.tick_params(labelsize=10)
# plt.xticks([0.1 * i for i in range(9, 1, -1)])
# label_font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
# plt.xlabel("correct label ratio", label_font)
# plt.ylabel("classification accuracy", label_font)
# plt.tight_layout()
# plt.savefig("accuracy.pdf")

# a = [0.8913, 0.802, 0.7121, 0.6229, 0.5254, 0.4302, 0.3318, 0.2333]
# ce = []
# from math import log
# def compute_ce(a):
#     b = (1 - a)
#     return - (a * log(a) + b * log(b))
# for alpha in a:
#     ce.append(compute_ce(alpha))
# print(ce)


model_x_data = [0.8913, 0.8020, 0.7121, 0.6229, 0.5254, 0.4302, 0.3318, 0.2333]
uniform_x_data = [0.8919, 0.7989, 0.7122, 0.6213, 0.5262, 0.4274, 0.33, 0.2354]
fix_data = [0.8920, 0.8054, 0.7122, 0.6248, 0.5264, 0.4275, 0.3326, 0.2335]
fix_data2 = [0.8054, 0.6248, 0.2335]

model_y_data_ce = [0.4298, 0.6744, 0.8487, 0.9876, 1.0924, 1.1692, 1.2106, 1.2070]
model_y_data_q = [0.3809, 0.5987, 0.7619, 0.8842, 0.9705, 1.0216, 1.0352, 1.0044]
uniform_y_data_ce = [0.6576, 1.0047, 1.2946, 1.5419, 1.7815, 1.9763, 2.1316, 2.2482]
uniform_y_data_q = [0.5826, 0.9327, 1.2328, 1.4912, 1.7347, 1.9353, 2.1036, 2.2278]
fix_y_data_ce = [0.5618, 0.7252, 0.6043]
fix_y_data_q = [0.3438, 0.4976, 0.6003, 0.6626, 0.6919, 0.6833, 0.6354, 0.5432]
ld = 1
ms = 10
figure = plt.figure()
plt.gca().invert_xaxis()
plt.plot(uniform_x_data, uniform_y_data_ce, 'b^-', label="UN", linewidth=ld, markersize=ms)
plt.plot(uniform_x_data, uniform_y_data_q, 'bs--', label="UN minimum", linewidth=ld, markersize=ms)
plt.plot(fix_data2, fix_y_data_ce, 'r^-', label="BN", linewidth=ld, markersize=ms)
plt.plot(fix_data, fix_y_data_q, 'rs--', label="BN minimum", linewidth=ld, markersize=ms)
plt.plot(model_x_data, model_y_data_ce, 'g^-', label="GN", linewidth=ld, markersize=ms)
plt.plot(model_x_data, model_y_data_q, 'gs--', label="GN minimum", linewidth=ld, markersize=ms)
legend_font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 13}
plt.legend(prop=legend_font)
plt.tick_params(labelsize=10)
plt.xticks([0.1 * i for i in range(9, 1, -1)])
label_font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
plt.xlabel("correct label ratio", label_font)
plt.ylabel("average of cross entropy", label_font)
plt.tight_layout()
plt.savefig("ce_loss.pdf")
