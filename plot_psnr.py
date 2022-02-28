from matplotlib import pyplot as plt

noise_ratio = [
    0,
    1,
    2.5,
    5,
    10,
    15,
    20,
    30,
]

psnr = [
    48.48,
    46.27,
    44.14,
    42.05,
    38.86,
    37.96,
    36.80,
    34.91,
]

fig = plt.figure()
plt.plot(noise_ratio, psnr)
plt.xlabel("Noise Level (%)")
plt.ylabel("PSNR (dB)")
# plt.xlim([0, 30])
plt.ylim([0, 50])
plt.title("Noise Reduction Performance")
plt.grid(True)
plt.show()
