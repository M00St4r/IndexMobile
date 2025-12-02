# IndexMobile
Control a Raspi Car Robot with gestures from your Webcam

# Setup

## 1. Connect to the Raspberry running GoPiGo
- Look for the Accesspoint and Connect to it.
- when your connected open your browser and open connect to gopigo.com

## 2. Setup the Code
- `gestures.py` will run on your PC but needs some libraries to be installed:
    - numpy
    - opencv
    - mediapipe
    - matplotlib
    - socket
    - json
- paste the `gestures_car.py` file to a custom directory on the Raspberry Pi
- you need to paste the **json library** into that directory get it [here](https://pypi.org/project/jsonlib/#files) and unpack it (you might need to unpack it twice)
- Now you need to edit line **3** in `gestures_car.py` to point to the correct path where you pasted the json library `sys.path.append("your_custom_path/jsonlib-1.6.1")`

## 3. Run the Code on the Robot
### option 1
- Open a console on your PC and run `ssh pi@your_pis_ip`
- you might need to enter a Password, you can try **raspberry** or **robots1234** if you don't know the password
- use `cd` to navigate your `gestures_car.py` file, could look something like that `cd /home/jupyter/guestureControl`
- run the code with `python gesture_car.py`
### option 2
- In the Web interface of the GoPiGo, open a Terminal
- navigate to your directory with `cd /home/jupyter/your_directory`
- run the code with `python gesture_car.py`

## 4. Run the Code on your PC
- navigate to where your `gestures.py` file is located, open a console there
- run `python gestures.py`

## 5. Stop the code
- in the console running `gestures.py` press **crtl + c**

