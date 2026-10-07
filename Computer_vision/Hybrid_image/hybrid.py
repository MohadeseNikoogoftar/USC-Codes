import argparse
import os
import numpy as np
from PIL import Image


def cross_correlation_2d(image, kernel):
  
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    kh, kw = kernel.shape

    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError(
        )

    pad_h = kh // 2
    pad_w = kw // 2

  
    if image.ndim == 2:

        padded = np.pad(
            image,
            ((pad_h, pad_h), (pad_w, pad_w)),
            mode="constant"
        )

        h, w = image.shape

        output = np.zeros((h, w), dtype=np.float64)

        for y in range(h):
            for x in range(w):

                region = padded[
                    y:y + kh,
                    x:x + kw
                ]

                output[y, x] = np.sum(region * kernel)

        return output

    elif image.ndim == 3:

        h, w, channels = image.shape

        padded = np.pad(
            image,
            (
                (pad_h, pad_h),
                (pad_w, pad_w),
                (0, 0)
            ),
            mode="constant"
        )

        output = np.zeros(
            (h, w, channels),
            dtype=np.float64
        )

        for y in range(h):
            for x in range(w):

                region = padded[
                    y:y + kh,
                    x:x + kw,
                    :
                ]

                for c in range(channels):
                    output[y, x, c] = np.sum(
                        region[:, :, c] * kernel
                    )

        return output

    else:
        raise ValueError(
            "Image must be either 2D grayscale or 3D RGB."
        )


def convolve_2d(image, kernel):
    kernel = np.asarray(kernel, dtype=np.float64)

    flipped_kernel = np.flip(kernel)

    return cross_correlation_2d(
        image,
        flipped_kernel
    )


def gaussian_blur_kernel_2d(kernel_size, sigma):
    if kernel_size <= 0:
        raise ValueError(
            "kernel_size must be greater than zero."
        )

    if kernel_size % 2 == 0:
        raise ValueError(
            "kernel_size must be an odd number."
        )

    if sigma <= 0:
        raise ValueError(
            "sigma must be greater than zero."
        )

    radius = kernel_size // 2

    x = np.arange(-radius, radius + 1)
    y = np.arange(-radius, radius + 1)

    xx, yy = np.meshgrid(x, y)

    kernel = np.exp(
        -(
            xx ** 2 + yy ** 2
        ) / (2 * sigma ** 2)
    )

    kernel /= np.sum(kernel)

    return kernel


def low_pass(image, kernel_size, sigma):
    kernel = gaussian_blur_kernel_2d(
        kernel_size,
        sigma
    )

    return convolve_2d(
        image,
        kernel
    )


def high_pass(image, kernel_size, sigma):

    image = np.asarray(
        image,
        dtype=np.float64
    )

    low = low_pass(
        image,
        kernel_size,
        sigma
    )

    high = image - low

    return high


def load_image(path):

    image = Image.open(path).convert("RGB")

    return np.asarray(
        image,
        dtype=np.float64
    )


def save_image(image, path):

    image = np.clip(
        image,
        0,
        255
    )

    image = image.astype(
        np.uint8
    )

    Image.fromarray(image).save(path)


def create_hybrid_image(
    image_low,
    image_high,
    low_kernel_size=21,
    low_sigma=5.0,
    high_kernel_size=21,
    high_sigma=5.0,
    high_weight=1.0,
    low_weight=1.0
):

    if image_low.shape != image_high.shape:
        raise ValueError(
        )

    low = low_pass(
        image_low,
        low_kernel_size,
        low_sigma
    )

    high = high_pass(
        image_high,
        high_kernel_size,
        high_sigma
    )

    hybrid = (
        low_weight * low
        +
        high_weight * high
    )

    return hybrid, low, high


def main():

    parser = argparse.ArgumentParser(
        description="Create a Hybrid Image from two input images."
    )

    parser.add_argument(
        "image1",
        help="Image used for low-frequency information."
    )

    parser.add_argument(
        "image2",
        help="Image used for high-frequency information."
    )

    parser.add_argument(
        "--low-kernel",
        type=int,
        default=21,
        help="Kernel size for low-pass filter. Default: 21"
    )

    parser.add_argument(
        "--low-sigma",
        type=float,
        default=5.0,
        help="Sigma for low-pass filter. Default: 5.0"
    )

    parser.add_argument(
        "--high-kernel",
        type=int,
        default=21,
        help="Kernel size for high-pass filter. Default: 21"
    )

    parser.add_argument(
        "--high-sigma",
        type=float,
        default=5.0,
        help="Sigma for high-pass filter. Default: 5.0"
    )

    parser.add_argument(
        "--low-weight",
        type=float,
        default=1.0,
        help="Weight of low-frequency component. Default: 1.0"
    )

    parser.add_argument(
        "--high-weight",
        type=float,
        default=1.0,
        help="Weight of high-frequency component. Default: 1.0"
    )

    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory. Default: output"
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    image1 = load_image(
        args.image1
    )

    image2 = load_image(
        args.image2
    )

    print(
        f"Image 1 shape: {image1.shape}"
    )

    print(
        f"Image 2 shape: {image2.shape}"
    )


    if image1.shape != image2.shape:

        raise ValueError(
            "The two images must have exactly the same "
            "height, width and number of channels."
        )


    hybrid, low, high = create_hybrid_image(
        image_low=image1,
        image_high=image2,
        low_kernel_size=args.low_kernel,
        low_sigma=args.low_sigma,
        high_kernel_size=args.high_kernel,
        high_sigma=args.high_sigma,
        low_weight=args.low_weight,
        high_weight=args.high_weight
    )

    left_path = os.path.join(
        args.output_dir,
        "left.png"
    )

    low_path = os.path.join(
        args.output_dir,
        "low_pass.png"
    )

    high_path = os.path.join(
        args.output_dir,
        "high_pass.png"
    )

    hybrid_path = os.path.join(
        args.output_dir,
        "hybrid.png"
    )

    save_image(
        image1,
        left_path
    )

    save_image(
        low,
        low_path
    )

    high_visual = high - np.min(high)

    max_value = np.max(high_visual)

    if max_value > 0:
        high_visual = (
            high_visual / max_value
        ) * 255

    save_image(
        high_visual,
        high_path
    )

    save_image(
        hybrid,
        hybrid_path
    )


    print()
    print("Hybrid image created successfully!")

    print(
        f"Low-pass kernel : "
        f"{args.low_kernel} x {args.low_kernel}"
    )

    print(
        f"Low-pass sigma  : "
        f"{args.low_sigma}"
    )

    print(
        f"High-pass kernel: "
        f"{args.high_kernel} x {args.high_kernel}"
    )

    print(
        f"High-pass sigma : "
        f"{args.high_sigma}"
    )

    print()
    print(
        f"Saved: {left_path}"
    )

    print(
        f"Saved: {low_path}"
    )

    print(
        f"Saved: {high_path}"
    )

    print(
        f"Saved: {hybrid_path}"
    )


if __name__ == "__main__":
    main()