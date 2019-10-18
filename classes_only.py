import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
from matplotlib.widgets import Slider, Button, RadioButtons
### A matplotlib figure is created with Simple_dog
### reference to this figure is cut when instance.close() is called
from matplotlib.patches import Patch
import os

class Grid_plot:
  
  def __init__(self):
    w_grid = 0.05
    h_grid = 0.05
    xedge_grid = 0.2
    yedge_grid = 0.7

    self.figure = plt.figure(figsize=(6,6))
    self.gridA1 = plt.axes([xedge_grid,yedge_grid,w_grid,h_grid])
    self.gridA2 = plt.axes([xedge_grid + 1*w_grid,yedge_grid,w_grid,h_grid])
    self.gridA3 = plt.axes([xedge_grid + 2*w_grid,yedge_grid,w_grid,h_grid])
    self.gridA4 = plt.axes([xedge_grid + 3*w_grid,yedge_grid,w_grid,h_grid])
    self.gridA5 = plt.axes([xedge_grid + 4*w_grid,yedge_grid,w_grid,h_grid])
    self.gridA6 = plt.axes([xedge_grid + 5*w_grid,yedge_grid,w_grid,h_grid])
    self.gridA7 = plt.axes([xedge_grid + 6*w_grid,yedge_grid,w_grid,h_grid])
    self.gridA8 = plt.axes([xedge_grid + 7*w_grid,yedge_grid,w_grid,h_grid])
    self.gridA9 = plt.axes([xedge_grid + 8*w_grid,yedge_grid,w_grid,h_grid])
    self.gridA10 = plt.axes([xedge_grid + 9*w_grid,yedge_grid,w_grid,h_grid])
    self.gridA11 = plt.axes([xedge_grid + 10*w_grid,yedge_grid,w_grid,h_grid])
    self.gridA12 = plt.axes([xedge_grid + 11*w_grid,yedge_grid,w_grid,h_grid])

    self.gridB1 = plt.axes([xedge_grid,yedge_grid-1*h_grid,w_grid,h_grid])
    self.gridB2 = plt.axes([xedge_grid + 1*w_grid,yedge_grid - 1*h_grid,w_grid,h_grid])
    self.gridB3 = plt.axes([xedge_grid + 2*w_grid,yedge_grid - 1*h_grid,w_grid,h_grid])
    self.gridB4 = plt.axes([xedge_grid + 3*w_grid,yedge_grid - 1*h_grid,w_grid,h_grid])
    self.gridB5 = plt.axes([xedge_grid + 4*w_grid,yedge_grid - 1*h_grid,w_grid,h_grid])
    self.gridB6 = plt.axes([xedge_grid + 5*w_grid,yedge_grid - 1*h_grid,w_grid,h_grid])
    self.gridB7 = plt.axes([xedge_grid + 6*w_grid,yedge_grid - 1*h_grid,w_grid,h_grid])
    self.gridB8 = plt.axes([xedge_grid + 7*w_grid,yedge_grid - 1*h_grid,w_grid,h_grid])
    self.gridB9 = plt.axes([xedge_grid + 8*w_grid,yedge_grid - 1*h_grid,w_grid,h_grid])
    self.gridB10 = plt.axes([xedge_grid + 9*w_grid,yedge_grid - 1*h_grid,w_grid,h_grid])
    self.gridB11 = plt.axes([xedge_grid + 10*w_grid,yedge_grid - 1*h_grid,w_grid,h_grid])
    self.gridB12 = plt.axes([xedge_grid + 11*w_grid,yedge_grid - 1*h_grid,w_grid,h_grid])

    self.gridC1 = plt.axes([xedge_grid,yedge_grid-2*h_grid,w_grid,h_grid])
    self.gridC2 = plt.axes([xedge_grid + 1*w_grid,yedge_grid - 2*h_grid,w_grid,h_grid])
    self.gridC3 = plt.axes([xedge_grid + 2*w_grid,yedge_grid - 2*h_grid,w_grid,h_grid])
    self.gridC4 = plt.axes([xedge_grid + 3*w_grid,yedge_grid - 2*h_grid,w_grid,h_grid])
    self.gridC5 = plt.axes([xedge_grid + 4*w_grid,yedge_grid - 2*h_grid,w_grid,h_grid])
    self.gridC6 = plt.axes([xedge_grid + 5*w_grid,yedge_grid - 2*h_grid,w_grid,h_grid])
    self.gridC7 = plt.axes([xedge_grid + 6*w_grid,yedge_grid - 2*h_grid,w_grid,h_grid])
    self.gridC8 = plt.axes([xedge_grid + 7*w_grid,yedge_grid - 2*h_grid,w_grid,h_grid])
    self.gridC9 = plt.axes([xedge_grid + 8*w_grid,yedge_grid - 2*h_grid,w_grid,h_grid])
    self.gridC10 = plt.axes([xedge_grid + 9*w_grid,yedge_grid - 2*h_grid,w_grid,h_grid])
    self.gridC11 = plt.axes([xedge_grid + 10*w_grid,yedge_grid - 2*h_grid,w_grid,h_grid])
    self.gridC12 = plt.axes([xedge_grid + 11*w_grid,yedge_grid - 2*h_grid,w_grid,h_grid])

    self.gridD1 = plt.axes([xedge_grid,yedge_grid-3*h_grid,w_grid,h_grid])
    self.gridD2 = plt.axes([xedge_grid + 1*w_grid,yedge_grid - 3*h_grid,w_grid,h_grid])
    self.gridD3 = plt.axes([xedge_grid + 2*w_grid,yedge_grid - 3*h_grid,w_grid,h_grid])
    self.gridD4 = plt.axes([xedge_grid + 3*w_grid,yedge_grid - 3*h_grid,w_grid,h_grid])
    self.gridD5 = plt.axes([xedge_grid + 4*w_grid,yedge_grid - 3*h_grid,w_grid,h_grid])
    self.gridD6 = plt.axes([xedge_grid + 5*w_grid,yedge_grid - 3*h_grid,w_grid,h_grid])
    self.gridD7 = plt.axes([xedge_grid + 6*w_grid,yedge_grid - 3*h_grid,w_grid,h_grid])
    self.gridD8 = plt.axes([xedge_grid + 7*w_grid,yedge_grid - 3*h_grid,w_grid,h_grid])
    self.gridD9 = plt.axes([xedge_grid + 8*w_grid,yedge_grid - 3*h_grid,w_grid,h_grid])
    self.gridD10 = plt.axes([xedge_grid + 9*w_grid,yedge_grid - 3*h_grid,w_grid,h_grid])
    self.gridD11 = plt.axes([xedge_grid + 10*w_grid,yedge_grid - 3*h_grid,w_grid,h_grid])
    self.gridD12 = plt.axes([xedge_grid + 11*w_grid,yedge_grid - 3*h_grid,w_grid,h_grid])

    self.gridE1 = plt.axes([xedge_grid,yedge_grid-4*h_grid,w_grid,h_grid])
    self.gridE2 = plt.axes([xedge_grid + 1*w_grid,yedge_grid - 4*h_grid,w_grid,h_grid])
    self.gridE3 = plt.axes([xedge_grid + 2*w_grid,yedge_grid - 4*h_grid,w_grid,h_grid])
    self.gridE4 = plt.axes([xedge_grid + 3*w_grid,yedge_grid - 4*h_grid,w_grid,h_grid])
    self.gridE5 = plt.axes([xedge_grid + 4*w_grid,yedge_grid - 4*h_grid,w_grid,h_grid])
    self.gridE6 = plt.axes([xedge_grid + 5*w_grid,yedge_grid - 4*h_grid,w_grid,h_grid])
    self.gridE7 = plt.axes([xedge_grid + 6*w_grid,yedge_grid - 4*h_grid,w_grid,h_grid])
    self.gridE8 = plt.axes([xedge_grid + 7*w_grid,yedge_grid - 4*h_grid,w_grid,h_grid])
    self.gridE9 = plt.axes([xedge_grid + 8*w_grid,yedge_grid - 4*h_grid,w_grid,h_grid])
    self.gridE10 = plt.axes([xedge_grid + 9*w_grid,yedge_grid - 4*h_grid,w_grid,h_grid])
    self.gridE11 = plt.axes([xedge_grid + 10*w_grid,yedge_grid - 4*h_grid,w_grid,h_grid])
    self.gridE12 = plt.axes([xedge_grid + 11*w_grid,yedge_grid - 4*h_grid,w_grid,h_grid])

    self.gridF1 = plt.axes([xedge_grid,yedge_grid-5*h_grid,w_grid,h_grid])
    self.gridF2 = plt.axes([xedge_grid + 1*w_grid,yedge_grid - 5*h_grid,w_grid,h_grid])
    self.gridF3 = plt.axes([xedge_grid + 2*w_grid,yedge_grid - 5*h_grid,w_grid,h_grid])
    self.gridF4 = plt.axes([xedge_grid + 3*w_grid,yedge_grid - 5*h_grid,w_grid,h_grid])
    self.gridF5 = plt.axes([xedge_grid + 4*w_grid,yedge_grid - 5*h_grid,w_grid,h_grid])
    self.gridF6 = plt.axes([xedge_grid + 5*w_grid,yedge_grid - 5*h_grid,w_grid,h_grid])
    self.gridF7 = plt.axes([xedge_grid + 6*w_grid,yedge_grid - 5*h_grid,w_grid,h_grid])
    self.gridF8 = plt.axes([xedge_grid + 7*w_grid,yedge_grid - 5*h_grid,w_grid,h_grid])
    self.gridF9 = plt.axes([xedge_grid + 8*w_grid,yedge_grid - 5*h_grid,w_grid,h_grid])
    self.gridF10 = plt.axes([xedge_grid + 9*w_grid,yedge_grid - 5*h_grid,w_grid,h_grid])
    self.gridF11 = plt.axes([xedge_grid + 10*w_grid,yedge_grid - 5*h_grid,w_grid,h_grid])
    self.gridF12 = plt.axes([xedge_grid + 11*w_grid,yedge_grid - 5*h_grid,w_grid,h_grid])

    self.gridG1 = plt.axes([xedge_grid,yedge_grid-6*h_grid,w_grid,h_grid])
    self.gridG2 = plt.axes([xedge_grid + 1*w_grid,yedge_grid - 6*h_grid,w_grid,h_grid])
    self.gridG3 = plt.axes([xedge_grid + 2*w_grid,yedge_grid - 6*h_grid,w_grid,h_grid])
    self.gridG4 = plt.axes([xedge_grid + 3*w_grid,yedge_grid - 6*h_grid,w_grid,h_grid])
    self.gridG5 = plt.axes([xedge_grid + 4*w_grid,yedge_grid - 6*h_grid,w_grid,h_grid])
    self.gridG6 = plt.axes([xedge_grid + 5*w_grid,yedge_grid - 6*h_grid,w_grid,h_grid])
    self.gridG7 = plt.axes([xedge_grid + 6*w_grid,yedge_grid - 6*h_grid,w_grid,h_grid])
    self.gridG8 = plt.axes([xedge_grid + 7*w_grid,yedge_grid - 6*h_grid,w_grid,h_grid])
    self.gridG9 = plt.axes([xedge_grid + 8*w_grid,yedge_grid - 6*h_grid,w_grid,h_grid])
    self.gridG10 = plt.axes([xedge_grid + 9*w_grid,yedge_grid - 6*h_grid,w_grid,h_grid])
    self.gridG11 = plt.axes([xedge_grid + 10*w_grid,yedge_grid - 6*h_grid,w_grid,h_grid])
    self.gridG12 = plt.axes([xedge_grid + 11*w_grid,yedge_grid - 6*h_grid,w_grid,h_grid])

    self.gridH1 = plt.axes([xedge_grid,yedge_grid-7*h_grid,w_grid,h_grid])
    self.gridH2 = plt.axes([xedge_grid + 1*w_grid,yedge_grid - 7*h_grid,w_grid,h_grid])
    self.gridH3 = plt.axes([xedge_grid + 2*w_grid,yedge_grid - 7*h_grid,w_grid,h_grid])
    self.gridH4 = plt.axes([xedge_grid + 3*w_grid,yedge_grid - 7*h_grid,w_grid,h_grid])
    self.gridH5 = plt.axes([xedge_grid + 4*w_grid,yedge_grid - 7*h_grid,w_grid,h_grid])
    self.gridH6 = plt.axes([xedge_grid + 5*w_grid,yedge_grid - 7*h_grid,w_grid,h_grid])
    self.gridH7 = plt.axes([xedge_grid + 6*w_grid,yedge_grid - 7*h_grid,w_grid,h_grid])
    self.gridH8 = plt.axes([xedge_grid + 7*w_grid,yedge_grid - 7*h_grid,w_grid,h_grid])
    self.gridH9 = plt.axes([xedge_grid + 8*w_grid,yedge_grid - 7*h_grid,w_grid,h_grid])
    self.gridH10 = plt.axes([xedge_grid + 9*w_grid,yedge_grid - 7*h_grid,w_grid,h_grid])
    self.gridH11 = plt.axes([xedge_grid + 10*w_grid,yedge_grid - 7*h_grid,w_grid,h_grid])
    self.gridH12 = plt.axes([xedge_grid + 11*w_grid,yedge_grid - 7*h_grid,w_grid,h_grid])#Code to make gridlist
    gridlist = [self.gridA1, self.gridA2, self.gridA3, self.gridA4, self.gridA5, self.gridA6, self.gridA7, self.gridA8, self.gridA9, self.gridA10, self.gridA11, self.gridA12,\
          self.gridB1, self.gridB2, self.gridB3, self.gridB4, self.gridB5, self.gridB6, self.gridB7, self.gridB8, self.gridB9, self.gridB10, self.gridB11, self.gridB12,\
            self.gridC1, self.gridC2, self.gridC3, self.gridC4, self.gridC5, self.gridC6, self.gridC7, self.gridC8, self.gridC9, self.gridC10, self.gridC11, self.gridC12,\
            self.gridD1, self.gridD2, self.gridD3, self.gridD4, self.gridD5, self.gridD6, self.gridD7, self.gridD8, self.gridD9, self.gridD10, self.gridD11, self.gridD12,\
            self.gridE1, self.gridE2, self.gridE3, self.gridE4, self.gridE5, self.gridE6, self.gridE7, self.gridE8, self.gridE9, self.gridE10, self.gridE11, self.gridE12,\
            self.gridF1, self.gridF2, self.gridF3, self.gridF4, self.gridF5, self.gridF6, self.gridF7, self.gridF8, self.gridF9, self.gridF10, self.gridF11, self.gridF12,\
            self.gridG1, self.gridG2, self.gridG3, self.gridG4, self.gridG5, self.gridG6, self.gridG7, self.gridG8, self.gridG9, self.gridG10, self.gridG11, self.gridG12,\
            self.gridH1, self.gridH2, self.gridH3, self.gridH4, self.gridH5, self.gridH6, self.gridH7, self.gridH8, self.gridH9, self.gridH10, self.gridH11, self.gridH12]
    self.axis_off(gridlist)
    #gridlist = [self.gridA1,self.gridA2,self.gridA3]
    #self.axis_off(gridlist)

    # Code that makes the grid buttons
    self.bgA1 = Button(self.gridA1, 'A1')
    self.bgA2 = Button(self.gridA2, 'A2')
    self.bgA3 = Button(self.gridA3, 'A3')
    self.bgA4 = Button(self.gridA4, 'A4')
    self.bgA5 = Button(self.gridA5, 'A5')
    self.bgA6 = Button(self.gridA6, 'A6')
    self.bgA7 = Button(self.gridA7, 'A7')
    self.bgA8 = Button(self.gridA8, 'A8')
    self.bgA9 = Button(self.gridA9, 'A9')
    self.bgA10 = Button(self.gridA10, 'A10')
    self.bgA11 = Button(self.gridA11, 'A11')
    self.bgA12 = Button(self.gridA12, 'A12')
    self.bgB1 = Button(self.gridB1, 'B1')
    self.bgB2 = Button(self.gridB2, 'B2')
    self.bgB3 = Button(self.gridB3, 'B3')
    self.bgB4 = Button(self.gridB4, 'B4')
    self.bgB5 = Button(self.gridB5, 'B5')
    self.bgB6 = Button(self.gridB6, 'B6')
    self.bgB7 = Button(self.gridB7, 'B7')
    self.bgB8 = Button(self.gridB8, 'B8')
    self.bgB9 = Button(self.gridB9, 'B9')
    self.bgB10 = Button(self.gridB10, 'B10')
    self.bgB11 = Button(self.gridB11, 'B11')
    self.bgB12 = Button(self.gridB12, 'B12')
    self.bgC1 = Button(self.gridC1, 'C1')
    self.bgC2 = Button(self.gridC2, 'C2')
    self.bgC3 = Button(self.gridC3, 'C3')
    self.bgC4 = Button(self.gridC4, 'C4')
    self.bgC5 = Button(self.gridC5, 'C5')
    self.bgC6 = Button(self.gridC6, 'C6')
    self.bgC7 = Button(self.gridC7, 'C7')
    self.bgC8 = Button(self.gridC8, 'C8')
    self.bgC9 = Button(self.gridC9, 'C9')
    self.bgC10 = Button(self.gridC10, 'C10')
    self.bgC11 = Button(self.gridC11, 'C11')
    self.bgC12 = Button(self.gridC12, 'C12')
    self.bgD1 = Button(self.gridD1, 'D1')
    self.bgD2 = Button(self.gridD2, 'D2')
    self.bgD3 = Button(self.gridD3, 'D3')
    self.bgD4 = Button(self.gridD4, 'D4')
    self.bgD5 = Button(self.gridD5, 'D5')
    self.bgD6 = Button(self.gridD6, 'D6')
    self.bgD7 = Button(self.gridD7, 'D7')
    self.bgD8 = Button(self.gridD8, 'D8')
    self.bgD9 = Button(self.gridD9, 'D9')
    self.bgD10 = Button(self.gridD10, 'D10')
    self.bgD11 = Button(self.gridD11, 'D11')
    self.bgD12 = Button(self.gridD12, 'D12')
    self.bgE1 = Button(self.gridE1, 'E1')
    self.bgE2 = Button(self.gridE2, 'E2')
    self.bgE3 = Button(self.gridE3, 'E3')
    self.bgE4 = Button(self.gridE4, 'E4')
    self.bgE5 = Button(self.gridE5, 'E5')
    self.bgE6 = Button(self.gridE6, 'E6')
    self.bgE7 = Button(self.gridE7, 'E7')
    self.bgE8 = Button(self.gridE8, 'E8')
    self.bgE9 = Button(self.gridE9, 'E9')
    self.bgE10 = Button(self.gridE10, 'E10')
    self.bgE11 = Button(self.gridE11, 'E11')
    self.bgE12 = Button(self.gridE12, 'E12')
    self.bgF1 = Button(self.gridF1, 'F1')
    self.bgF2 = Button(self.gridF2, 'F2')
    self.bgF3 = Button(self.gridF3, 'F3')
    self.bgF4 = Button(self.gridF4, 'F4')
    self.bgF5 = Button(self.gridF5, 'F5')
    self.bgF6 = Button(self.gridF6, 'F6')
    self.bgF7 = Button(self.gridF7, 'F7')
    self.bgF8 = Button(self.gridF8, 'F8')
    self.bgF9 = Button(self.gridF9, 'F9')
    self.bgF10 = Button(self.gridF10, 'F10')
    self.bgF11 = Button(self.gridF11, 'F11')
    self.bgF12 = Button(self.gridF12, 'F12')
    self.bgG1 = Button(self.gridG1, 'G1')
    self.bgG2 = Button(self.gridG2, 'G2')
    self.bgG3 = Button(self.gridG3, 'G3')
    self.bgG4 = Button(self.gridG4, 'G4')
    self.bgG5 = Button(self.gridG5, 'G5')
    self.bgG6 = Button(self.gridG6, 'G6')
    self.bgG7 = Button(self.gridG7, 'G7')
    self.bgG8 = Button(self.gridG8, 'G8')
    self.bgG9 = Button(self.gridG9, 'G9')
    self.bgG10 = Button(self.gridG10, 'G10')
    self.bgG11 = Button(self.gridG11, 'G11')
    self.bgG12 = Button(self.gridG12, 'G12')
    self.bgH1 = Button(self.gridH1, 'H1')
    self.bgH2 = Button(self.gridH2, 'H2')
    self.bgH3 = Button(self.gridH3, 'H3')
    self.bgH4 = Button(self.gridH4, 'H4')
    self.bgH5 = Button(self.gridH5, 'H5')
    self.bgH6 = Button(self.gridH6, 'H6')
    self.bgH7 = Button(self.gridH7, 'H7')
    self.bgH8 = Button(self.gridH8, 'H8')
    self.bgH9 = Button(self.gridH9, 'H9')
    self.bgH10 = Button(self.gridH10, 'H10')
    self.bgH11 = Button(self.gridH11, 'H11')
    self.bgH12 = Button(self.gridH12, 'H12')

    plt.show(self.figure)

  def axis_off(self,axislist):
    for axis in axislist:
      axis.set_xticks([])
      axis.set_yticks([])
      axis.set_xticklabels([])
      axis.set_yticklabels([])
  
class MyCircle(mpl.patches.Ellipse):
    """
    A circle patch.
    """
    def __str__(self):
        pars = self.center[0], self.center[1], self.radius
        fmt = "Circle(xy=(%g, %g), radius=%g)"
        return fmt % pars

    def __init__(self, xy, dog_instance, radius=5,**kwargs):
        """
        Create true circle at center *xy* = (*x*, *y*) with given
        *radius*.  Unlike :class:`~matplotlib.patches.CirclePolygon`
        which is a polygonal approximation, this uses Bezier splines
        and is much closer to a scale-free circle.

        Valid kwargs are:
        %(Patch)s

        """
        mpl.patches.Ellipse.__init__(self, xy, radius * 2, radius * 2, **kwargs)
        self.radius = radius
        self._center = xy
        self.instance = dog_instance
        
    def set_radius(self, radius):
        """
        Set the radius of the circle

        Parameters
        ----------
        radius : float
        """
        self.width = self.height = 2 * radius
        self.stale = True

    def get_radius(self):
        """
        Return the radius of the circle
        """
        return self.width / 2.

    def set_center(self, xy):
        """
        Set the center of the ellipse.

        Parameters
        ----------
        xy : (float, float)
        """
        self._center = xy
        self.stale = True

    def get_center(self):
        """
        Return the center of the ellipse
        """
        return self._center

    radius = property(get_radius, set_radius)
    center = property(get_center, set_center)



class CircleMover:
    def __init__(self, line, circle, circle_center, inner_circle, tcenter, textbox):
        #print('CircleMover')
        self.line = line
        #print('line is :'+str(type(self.line)))
        self.xs = list(line.get_xdata())
        #print('self.xs is:' + str(type(self.xs)))
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.circle = circle
        self.center = circle_center
        self.inner = inner_circle
        self.text = textbox
        self.tcenter = tcenter

    def __call__(self, event):
        print('click', event)
        if event.inaxes != self.line.axes: return
        #self.xs.append(event.xdata) #each time a button is clicked the x and y coordinates are saved in the matplotlib.backend_bases.MouseEvent
        #self.ys.append(event.ydata)
        #self.line.set_data(self.xs,self.ys) #function of matplotlib.lines
        well_obj = self.circle.instance.well_obj
        self.circle.set_center((event.xdata,event.ydata))
        self.center.set_center((event.xdata,event.ydata))
        self.inner.set_center((event.xdata,event.ydata))
        self.tcenter.set_center((event.xdata,event.ydata))

        #well_obj.update_wcx(event.xdata)
        #well_obj.update_wcy(event.ydata)
        #well_obj.update_wr(well_obj.wr)

        well_obj.update_tx(event.xdata)
        well_obj.update_ty(event.ydata)

        print('Sx = 1')
        print('call :' + str(self))
        self.circle.instance.update_data_onclick(event)
        sx,sy = self.circle.instance.print_updated_displacement(event)

        well_obj.sx = sx
        well_obj.sy = sy

        plt.draw()


class Index(object):
    ind = 0

    def next(self, event):
        self.ind += 1
        print(self.ind)
        print(object)

    def prev(self, event):
        self.ind -= 1
        print(self.ind)


class Well_well_well():
  def __init__(self, wellcenterx,wellcentery,wellcenterradii, dropcenterx, dropcentery, dropradii, targetcenterx, targetcentery, targetradii, name,impath,sx,sy,volume):
    #parameters a well needs
    #an offset, everything for plotting, a name
    self.name = name
    self.wcx = wellcenterx
    self.wcy = wellcentery
    self.wr = wellcenterradii

    self.dcx = dropcenterx
    self.dcy = dropcentery
    self.dr = dropradii

    self.tx = targetcenterx
    self.ty = targetcentery
    self.tr = targetradii

    self.im_path = impath

    self.sx = 0
    self.sy = 0
    self.volume = volume
    #Parameters to hold displacements in microns


  def update_wcx(self,update):
    self.wcx = update
  def update_wcy(self,update):
    self.wcy = update
  def update_wr(self,update):
    self.wr = update
  def update_dcx(self,update):
    self.dcx = update
  def update_dcy(self,update):
    self.dcy = update
  def update_dr(self,update):
    self.dr = update
  def update_tx(self,update):
    self.tx = update
  def update_ty(self,update):
    self.ty = update
  def update_tr(self,update):
    self.dr = update    
  def update_sx(self,update):
    self.sx = update
  def update_sy(self,update):
    self.sy = update
  def update_volume(self,event):
    self.volume = event
    print(self.volume)
  def get_volume():
    volume = self.volume
    return volume


class Plate():
  well_names = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', \
              'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12', \
              'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', \
              'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09', 'D10', 'D11', 'D12', \
              'E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E08', 'E09', 'E10', 'E11', 'E12', \
              'F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10', 'F11', 'F12', \
              'G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11', 'G12', \
              'H01', 'H02', 'H03', 'H04', 'H05', 'H06', 'H07', 'H08', 'H09', 'H10', 'H11', 'H12', ]
  plate_dict = {}
  for well in well_names:
    #plate_dict[well]=Well_well_well(600,600,400,600,600,20,0,0,0,well,'test',0,0,125)
    plate_dict[well]=Well_well_well(600,600,400,600,600,20,0,0,0,well,'test',0,0,25)
  def __init__(self,plate_dict):
    self.plate_dict = plate_dict
  '''
  def __init__(self):
    self.wells = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', \
              'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', \
              'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', \
              'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', \
              'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', \
              'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', \
              'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', \
              'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', ]
    self.order = range(0,96)

    ### Params
      ### x-center, y-center, well-radii, drop-radii, x-target, y-target
    self.inital_params = ()
    self.updated_params = ()
    '''

class Simple_dog: #plotting class
  "A simple dog class"
  def __init__(self, well_obj):
    self.well_obj = well_obj
    self.bark = 'woof'

    self.wxpos = well_obj.wcx
    self.wypos = well_obj.wcy

    self.tx = well_obj.tx
    self.ty = well_obj.ty

    self.sx = well_obj.sx
    self.sy = well_obj.sy

    self.wr = well_obj.wr
    self.r_inner = 0.73701*self.wr
    self.r_inner = int(self.r_inner)
    #self.r_inner = self.r_inner.astype(int)

    self.dxpos = well_obj.dcx
    self.dypos = well_obj.dcy
    self.dr = well_obj.dr

    self.impath = well_obj.im_path

    self.figure = plt.figure(figsize=(16,8))


    self.x_slider_axes = plt.axes([0.08, 0.1, 0.6, 0.03])
    #self.y_slider_axes = plt.axes([0.05,0.05,0.6,0.03])
    #self.radii_slider_axes  = plt.axes([0.05, 0.15, 0.6, 0.03])

    ### Intiate Axes
    self.ax1= plt.axes([0.05,0.25,0.6,0.6])
    self.ax2= plt.axes([0.65,0.25,0.3,0.3])

    self.text = self.ax2.text(0,0,"I'm all the data")
    self.ax2.axis('off')
    self.text.set_text('tx,ty %s,%spx\nsx,sy %s,%sum' %(self.well_obj.tx,self.well_obj.ty,well_obj.sx,well_obj.sy))
    
    self.line, = self.ax1.plot([0],[0])

    self.im = mpimg.imread(self.impath)
    self.img = self.ax1.imshow(self.im)#,cmap='Greys_r')

    self.well_circle = self.ax1.add_artist(MyCircle((self.wxpos,self.wypos),self,self.wr, color='b',fill=False))
    self.inner_circle = self.ax1.add_artist(MyCircle((self.wxpos,self.wypos),self,self.r_inner,color='b', fill=False))
    self.well_center = self.ax1.add_artist(MyCircle((self.wxpos,self.wypos),self,2, color='b', fill=False))

    #self.CircleMover=CircleMover(self.line,self.well_circle,self.well_center, self.inner_circle,self.text)


    self.target_center = self.ax1.add_artist(MyCircle((self.tx,self.ty),self,2,color='r',fill=False))
    self.target_circle = self.ax1.add_artist(MyCircle((self.tx,self.ty),self,self.dr, color='g',fill=False))
    #self, line, circle, circle_center, inner_circle, tcenter, textbox

    self.CircleMover=CircleMover(self.line, self.target_circle, self.target_center, self.target_circle, self.target_center, self.text)
    #self.CircleMover=CircleMover(self.line, self.target_circle, self.target_center, self.target_circle, self.text)

    self.drop_circle = self.ax1.add_artist(MyCircle((self.dxpos,self.dypos),self,self.dr,color='r',fill=False))   
    self.drop_center = self.ax1.add_artist(MyCircle((self.dxpos,self.dypos),self,2,color='r',fill=False))
    


    self.ax3= plt.axes([0.8,0.8,0.3,0.3])
    #self.ax3.axis('off')
    #self.activestatus = 'Well Circle'
    #self.active_circle_button = Button(self.ax3, 'Active Circle: ' + self.activestatus)
    #self.active_circle_button.on_clicked(self.update_active_circle)
    


    #self.dataCID=self.figure.canvas.mpl_connect('button_press_event', self.update_data_onclick(self.text))
    
    self.xSlider = Slider(self.x_slider_axes, 'volume', 0, 300, valinit=well_obj.volume, valstep=25)
    #self.ySlider = Slider(self.y_slider_axes, 'wypos', 0, 200, valinit=100)
    #self.rSlider = Slider(self.radii_slider_axes, 'Radius', 0, 600, valinit=300)

    self.xCID = self.xSlider.on_changed(well_obj.update_volume)
    #self.xCID2 = self.xSlider.on_changed(self.f2)

    #self.yCID = self.ySlider.on_changed(self.f)
    #self.rCID = self.rSlider.on_changed(self.well_circle.set_radius)
    #self.r2CID = self.rSlider.on_changed(self.inner_circle.set_radius)


    self.callback = Index()
    self.axprev = plt.axes([0.76, 0.05, 0.1, 0.07])
    self.axnext = plt.axes([0.88, 0.05, 0.1, 0.07])
    self.ax3= plt.axes([0.5,0.6,0.3,0.3])

    impath = well_obj.im_path
    parent = os.path.dirname(impath)
    files = os.listdir(parent)
    droppath = parent + '/' + files[1]
    droppath2 = parent + '/' + files[0]

    drop = plt.imread(droppath)
    plt.imshow(drop)
    drop2 = plt.imread(droppath2)
    self.ax4 = plt.axes([0.7,0.6,0.3,0.3])
    plt.imshow(drop2)




    self.bnext = Button(self.axnext, 'Next')
    self.bnext.on_clicked(self.close)
    self.bprev = Button(self.axprev, 'Previous')
    self.bprev.on_clicked(self.close)

    self.update_title(well_obj.name)

    #grid axis
    
    
    #gridlist = [self.gridA1,self.gridA2,self.gridA3]
    #self.axis_off(gridlist)

    # Code that makes the grid buttons

    ### How CID's seem to work: on_changed and disconnect are called on an object
    ### That object has a CID per function it connects to starting with 0
    ### different objects can have the same CID
    ### This will connect/disconnect the changing of that object to calling the function

  def f(self,other):
    print('changing' + str(other) + '\n')
    print(self.xCID)
    print(self.yCID)
    print(self.rCID)
  def f2(self, other):
    print(self.xCID2)
  def updateX(self, update):
    self.wxpos = update
  def updateY(self, update):
    self.wxpos = update
  def updateBark(self,string):
    self.bark = string

  def update_title(self,text):
    self.ax1.set_title(text)
  def close(self, buttonpressevent):
    plt.close(self.figure)
    print(buttonpressevent) 
  def show(self):
    plt.draw()
    plt.show(self.figure)
  def update_data_onclick(self,event):

    xString = str(np.round(event.xdata, decimals=0))
    yString = str(np.round(event.ydata, decimals=0))
    string ='Well : X ' + xString + '  ' + 'Y ' + yString
    print(type(self))
    #self.text.set_text(string)

    print(string)
  def axis_off(self,axislist):
    for axis in axislist:
      axis.set_xticks([])
      axis.set_yticks([])
      axis.set_xticklabels([])
      axis.set_yticklabels([])


  #Development funciton
  def make_grid_list(self):
    gridlist = []
    for name in Plate.well_names:
      grid = 'self.grid' + name
      gridlist.append(grid)
    return gridlis
  def change_bundeled_radii(event):
    print('a')
  def update_active_circle(self,event):
    if self.activestatus == 'Well Circle':
      self.activestatus = 'Drop Circle'
    else:
      self.activestatus = 'Well Circle'
    self.ax3.set_title(self.activestatus)
    #dummy = self.ax3.text(0.5, 0.5, self.activestatus,verticalalignment='center',horizontalalignment='center',transform=self.ax3.transAxes)
    plt.draw()
  def print_updated_displacement(self,event):
    #well_inner = self.inner_circle.width[0]
    well_inner = self.inner_circle.width
    microns_per_px = 2770/well_inner
    #~print('microns per pixel' + microns_per_px)


    tx = event.xdata*microns_per_px
    ty = event.ydata*microns_per_px

    cx = self.well_circle._center[0]*microns_per_px
    cy = self.well_circle._center[1]*microns_per_px

    wx = tx-cx
    wy = cy-ty


    sx = np.round(wx[0], decimals = 0)
    sy = np.round(wy[0],decimals = 0)

    # add predicted offsets to hit the center of the well

    sx = sx -600 #um
    sy = sy + 180 #um

    self.well_obj.update_sx(sx)
    self.well_obj.update_sy(sy)

    dropx = self.drop_circle._center[0]*microns_per_px
    dropy = self.drop_circle._center[1]*microns_per_px
    dx = tx-dropx #um
    dy = dropy-ty #um

    mnwd = np.sqrt(wx**2+wy**2) #um
    mndd = np.sqrt(dx**2+dy**2) #um

    mnwd = mnwd[0]
    mndd = mndd[0]

    mnwd = np.round(mnwd, decimals=0)
    mndd = np.round(mndd, decimals=0)

    sdx = np.round(dx[0], decimals = 0)
    sdy = np.round(dy[0],decimals = 0)
    

    #2.77 mm well_inner 
    print(well_inner)

    string = '\n\ninner well diameter %s px\npixel size %s um\n\nWell Displacement:\nX: %s um\nY: %s um\nmean %s' %(well_inner,np.round(microns_per_px,decimals=2),sx, sy, mnwd)
    string2 = '\n\nDrop displacement:\nX: %s um\nY: %s um\nmean %s um' %(sdx,sdy,mndd)
    print(string)
    self.text.set_text(string+string2)
    return sx, sy

