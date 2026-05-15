from setuptools import find_packages, setup

package_name = "robot_vision"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    package_data={"robot_vision.web": ["static/*"]},
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", ["config/app.yaml"]),
    ],
    install_requires=[
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "numpy>=1.24",
        "pillow>=10",
        "pyyaml>=6",
    ],
    extras_require={
        "camera": ["opencv-python>=4.8"],
        "train": [
            "torch>=2.1",
            "torchvision>=0.16",
            "transformers>=4.40",
            "accelerate>=0.27",
        ],
        "test": ["pytest>=8", "httpx>=0.27"],
    },
    zip_safe=True,
    maintainer="Caleb Steinhauer",
    maintainer_email="csteinhauer@marvelsaws.com",
    description="Browser-based Astra depth camera part inspection app for Jetson/ROS2 workspaces.",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "robot-vision-web=robot_vision.cli:web",
            "robot-vision-camera-check=robot_vision.cli:camera_check",
            "robot-vision-inspect-sample=robot_vision.cli:inspect_sample",
            "robot-vision-train-vision=robot_vision.cli:train_vision",
        ],
    },
)
