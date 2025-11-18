# UAVPath - A Computer Science Capstone Project

After years of honing our skills in software engineering, 3D graphics, and complex problem-solving, our team is proud to present our final year Computer Science Capstone Project: **UAVPath**.

## The Challenge

Our team was tasked with building upon an innovative concept that merges the worlds of technology and archaeology: a UAV flight path simulation tool. An initial program, developed by a PhD student, laid the groundwork for planning drone flights to create photogrammetric reconstructions of culturally significant sites.

However, this initial version required significant enhancements to become a truly interactive and visually intuitive tool. The core challenge was to transform a functional prototype into a polished, user-friendly application that offered richer visualizations, deeper user interactivity, and more powerful analytical features for comparing flight paths.

## Our Solution: The Next Generation of UAVPath

To meet this challenge, we developed the next generation of **UAVPath** – a sophisticated flight path simulation and visualization platform. This application provides a seamless and visually compelling way to plan, simulate, and analyze drone flights around detailed 3D models.

It is built using Python, leveraging the power of libraries like Open3D for advanced 3D rendering and Qt Designer for a clean and responsive user interface.

### Key Features

We have developed a suite of features designed to enhance user satisfaction and provide powerful tools for researchers and operators:

*   **Advanced 3D Visualization:** The application renders high-definition 3D models in `.obj` and `.ply` formats, providing a realistic canvas for flight simulation. Users can interact smoothly with the models, inspecting sites from any angle.
*   **Intuitive Flight Path Analysis:** Go beyond simple lines on a screen. Flight paths are rendered with enhanced visuals, spline interpolation for smooth curves, and color-coordination, allowing for clear and easy comparison between different trajectories.
*   **Immersive First-Person View (FPV):** Step into the drone's cockpit. At any waypoint, the user can instantly switch to an FPV, showing the precise image the drone would capture from that location and angle, allowing for meticulous shot planning.
*   **Flight Simulation and Video Export:** The system animates the drone's complete path around the 3D model, and this entire simulation can be recorded and exported as a video file for presentations, record-keeping, or further analysis.

## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

This application is built with Python. You will need to have Python and the `pip` package manager installed.

### Installation

1.  **Clone the repository**
    ```sh
    git clone https://gitlab.cs.uct.ac.za/ram_uavpath/Capstone
    ```

    or (IF THE ABOVE DOESN'T EXIST)

    ```sh
    git clone https://github.com/Plum165/Capstone
    ```
2.  **Navigate to the project directory**
    ```sh
    cd Capstone
    ```
3.  **Install the required packages**
    ```sh
    pip install -r requirements.txt
    ```
    

## Usage

To launch the application, navigate to the project folder and run the main script:

```sh
python UAV_UI.py
```
Upon launching, the UAVpath Simulator window will appear. From there, you can log on and begin loading 3D models and their corresponding flight path data. For a detailed guide on using all the features, please refer to the User Manual in Appendix A of the project report.

## Team Members
This project was successfully developed and delivered by:

Aneesah Barnes - [Linkin] - <https://www.linkedin.com/in/aneesah-barnes-724143237/> 
Marcus Buxmann - NA
Rashaad Samsodien - [Linkin] - <https://www.linkedin.com/in/moegamatsamsodien/>