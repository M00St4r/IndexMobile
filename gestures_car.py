import socket
import sys
sys.path.append("/home/jupyter/guestureControl/jsonlib-1.6.1")
import json
import time
import easygopigo3 as easy

# /home/jupyter/guestureControll/gesture.py
my_gopigo = easy.EasyGoPiGo3()

serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversocket.bind(('10.10.10.10', 8089))
serversocket.listen(5) # become a server socket, maximum 5 connections

while True:
    connection, address = serversocket.accept()
    with connection:
        print(f"connected by {address}")
        
        while True:
            data = connection.recv(1024).decode('utf-8')
            print(data)
            
            try: 
                move = json.loads(data)
                print(move)
                direction = move.get("dir")

                if direction == "forward":
                    my_gopigo.set_speed(250)
                    my_gopigo.forward()
                elif direction == "backward":
                    my_gopigo.set_speed(250)
                    my_gopigo.backward()
                elif direction == "right":
                    my_gopigo.set_speed(100)
                    my_gopigo.spin_right()
                elif direction == "left":
                    my_gopigo.set_speed(100)
                    my_gopigo.spin_left()
                elif direction == "none":
                    my_gopigo.stop()
            except:
                print("wrong json")
            if data == 'close':
                serversocket.close()
                my_gopigo.stop()
                break