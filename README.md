This is a small project with a neural network that identifies objects in real time. To use it, you need to have a lot of well-processed 224 by 224 photos in .png format, sorted by class.

The file create_vision_model.py - just creates a model that can be used to define objects. It indicates the changes that need to be replaced.

The file object_identifier.py This is the main file that needs to be run for the neural network to work. 
It also specifies the variables that need to be changed to yours. A window with a camera will open, where the names of objects that the neural network was able to identify will be displayed.

The objects themselves are in no way singled out separately, only the name of the object is displayed. 
The file specifies a condition in which you can perform various tasks for further processing after the definition.
 
 ██████  ███    ███  ██ ███████ ████████  ██ ██   ██ ██████   ██████   ██████  
██    ██ ████  ████ ███ ██         ██    ███ ██  ██       ██ ██       ██  ████ 
██ ██ ██ ██ ████ ██  ██ ███████    ██     ██ █████    █████  ███████  ██ ██ ██ 
██ ██ ██ ██  ██  ██  ██      ██    ██     ██ ██  ██       ██ ██    ██ ████  ██ 
 █ ████  ██      ██  ██ ███████    ██     ██ ██   ██ ██████   ██████   ██████  
