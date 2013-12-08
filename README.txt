Libraries Required:

CUDA
ParMETIS/METIS
OpenGL
GLUT
GLEW


Runtime Operation:

Mouse 1: Rotate
Mouse 2: Zoom
Mouse 1+2: Translate

Shift + Mouse 1:
     Create draggable 'keynode' at nearest visible mesh node

Drag Keynode:
     Apply boundary forces

Ctrl + Mouse 1:
     Delete keynode

'c': Center
     Centers the view on the mesh

'r': Reset
     Returns the simulation to its initial state. Useful after an element inversion causes NaNs to wipeout the simulation

'p': Pause/Resume
     Pauses the simulation

esc: Exit
     Exits the simulation and closes the visualization window

'R': Record
     Saves each frame of the animation as a binary RBG file called ######.bin. WARNING: Large, uncompressed files and slow writes.

