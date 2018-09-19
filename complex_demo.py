import plotly
import plotly.graph_objs as go
import numpy as np
from plotly import tools
import matrixDFT
import copy

def get_TranslateMatrix(vector):
    return np.array([[1, 0, 0, vector[0]],
                     [0, 1, 0, vector[1]],
                     [0, 0, 1, vector[2]],
                     [0, 0, 0, 1]])

def get_RotationMatrix(fromMatrix, toMatrix):
    """Get the transformation matrix from fromMatrix to toMatrix."""
    rotation_Matrix = np.zeros((4, 4))
    rotation_Matrix[:3, :3] = toMatrix.dot(np.linalg.inv(fromMatrix))
    #rotation_Matrix[:, 3] = 1
    return rotation_Matrix

def transform(vertices, origin=np.array([0, 0, 0]), x_axis=None,
                        y_axis=None, z_axis=None):
    """Transform given vertices to another world space.
    @param vertices: vertices that should be transformed to new world.
                     numpy array with shape (n, 3)
    @param origin: the new worlds' origin point. numpy array with shape (3,)
    @param x_axis: x axis of the new world. numpy array with shape (3,)
    @param y_axis: y axis of the new world. numpy array with shape (3,)
    @param z_axis: z axis of the new world. numpy array with shape (3,)
    @return: vertices after the transformation with shape (n, 3) and the same
             order.
    """
    def generate_axis(a, b):
        # One of a/b should not be None
        rand_vector = np.random.rand(3)
        rand_vector /= np.linalg.norm(rand_vector)
        if a is None:
            a = np.cross(b, rand_vector)
        if b is None:
            b = np.cross(a, rand_vector)
        return np.cross(a, b), a, b
        
    if x_axis is None:
        x_axis, y_axis, z_axis = generate_axis(y_axis, z_axis)
    if y_axis is None:
        y_axis, x_axis, z_axis = generate_axis(x_axis, z_axis)
    if z_axis is None:
        z_axis, x_axis, y_axis = generate_axis(x_axis, y_axis)
    worldMatrix = np.concatenate((x_axis, y_axis, z_axis)).reshape(3, 3)

    # Find transformation matrix: translation*rotation
    translateMatrix = get_TranslateMatrix(origin)
    localMatrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    rotation_Matrix = get_RotationMatrix(localMatrix.T, worldMatrix.T)
    transform_Matrix = translateMatrix.dot(rotation_Matrix)

    vertices = np.insert(vertices, vertices.shape[0], 0, axis=0)
    vertices = np.insert(vertices, vertices.shape[1], 1, axis=1)
    vertices = transform_Matrix.dot(vertices.T).T[:, :-1]
    return vertices


def lineTo3DMesh(x_list, y_list):
    """Generate 3D mesh from the 2D line.
    @param line_function: Given x, get the corresponding y
    @param x_start: starting x value.
    @param x_end: ending x value.
    @param sample: number of sample point.
    @return: (vertices, triangles) numpy array pair with shape (2*sample, 3)
    """
    nv = 0 
    vertices = np.zeros((0, 3))
    triangles = np.zeros((0, 3))
    
    for i in range(len(x_list)):
        x = x_list[i]
        y = y_list[i]
        
        new_points = np.array(
            [[x, 0, 0], [x, y, 0]])
        vertices = np.append(vertices, new_points, axis=0)
        nv += 2
        if nv > 2:
            new_triangles = np.array(
                [[nv - 4, nv - 3, nv - 2], [nv - 3, nv - 1, nv - 2]])
            triangles = np.append(triangles, new_triangles, axis=0)    
    return vertices, triangles



def make1DGaussian(s, sigma, ctr=None, x0 = 0.):
    #normalization?

    if ctr is None:
        ctr = s/2.0 
    
    x = np.linspace(-ctr+0.5, s-ctr-0.5, s)
    
    deg = -np.pi/180.0    # minus sign seen here ... any reason?  anand

    array = np.exp(-(x-x0)**2./(2.*sigma**2.))
    return x, array


def add_linesurface(x_data, y_data, axis):
    vertices, triangles = lineTo3DMesh(x_data, y_data)

    if axis == 'y':
        worldVertices = transform(
        vertices, x_axis=np.array([1, 0, 0]), y_axis=np.array([0, 0, 1]))
    elif axis == 'z':
        worldVertices = transform(
        vertices, x_axis=np.array([1, 0, 0]), y_axis=np.array([0, 1, 0]))
    
    mesh = go.Mesh3d(x=worldVertices[:, 0], 
                     y=worldVertices[:, 1],
                     z=worldVertices[:, 2], 
                     i=triangles[:, 0], 
                     j=triangles[:, 1],
                     k=triangles[:, 2])

    return mesh

THRESHOLD = 10**14
def clean_value(complex_data):
    real_amp = np.sum(np.abs(complex_data.real))
    imag_amp = np.sum(np.abs(complex_data.imag))
    try:
        if real_amp > THRESHOLD * imag_amp:
            print 'imag part is tiny and set to 0'
            complex_data_copy = copy.deepcopy(complex_data)
            complex_data_copy.imag *= 0
            return complex_data_copy
        elif imag_amp > THRESHOLD * real_amp:
            print 'real part is tiny set to 0'
            complex_data_copy = copy.deepcopy(complex_data)
            complex_data_copy.real *= 0
            return complex_data_copy
        else:
            return complex_data
    except ValueError:
        return complex_data


def complex_3dplot(x_data, complex_data, real_color='blue', imag_color='red', line_color='cyan', **kwargs):


    #if np.iscomplex(complex_data[0]):
    complex_data = clean_value(complex_data)

    traces = []

    mesh = add_linesurface(x_data, complex_data.real, axis='y')
    mesh.update(opacity=0.8, color=real_color)
    traces.append(mesh)
    
    mesh = add_linesurface(x_data, complex_data.imag, axis='z')
    mesh.update(opacity=0.8, color=imag_color)
    traces.append(mesh)

    traces.append(go.Scatter3d(x=x_data, 
                               y=complex_data.imag, 
                               z=complex_data.real, mode="lines", 
        line=dict(
        color=line_color,
        width=3
    )))

    if kwargs.get('plot'):
        fig = go.Figure(data=traces, layout=go.Layout(
        showlegend=False,
        ))
        if kwargs.get('filename'):
            plotly.offline.plot(fig, filename=kwargs['filename'])
        else:
            plotly.offline.iplot(fig)
    
    if kwargs.get('return_traces'):
        return traces


def complex_3danimation(x_data, complex_data_list, changing_variable_list, real_color='blue', imag_color='red', line_color='cyan', **kwargs):
    figure = {
    'data': [],
    'layout': {'xaxis': {'range': [-40, 40], 'autorange': False},
               'yaxis': {'range': [-3, 3], 'autorange': False}
               },
    'frames': []
    }

    sliders_dict = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue':{
            'font': {'size': 20},
            'prefix': 'Shift:',
            'visible': True,
            'xanchor': 'right'
        },
        'transition':{
            'duration': 300,
            'easing': 'cubic-in-out'
        },
        'pad':{'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1, 
        'y': 0,
        'steps': []
    }
    layout = go.Layout(
        scene = dict(
        xaxis = dict(
            range = [-40,40],),
        yaxis = dict(
            range = [-3,3],),
        zaxis = dict(
            range = [-3,3],),
        ),
        width=700,
        margin=dict(
        r=20, l=10,
        b=10, t=10)
      )

    for i in range(len(changing_variable_list)):
        changing_var = changing_variable_list[i]
        frame = {'data': [], 'name': str(changing_var)}
        data_dict = complex_3dplot(x_data, complex_data_list[i], real_color=real_color, imag_color=imag_color, line_color=line_color, return_traces=True)

        if i == 0:
            figure['data'] = data_dict
            figure['layout'] = layout

        frame['data'] = data_dict
        frame['layout'] = layout
        figure['frames'].append(frame) 

        slider_step = {
            'args': [
                [changing_var],
                {'frame': 
                    {'duration': 300, 'redraw': False},
                    'mode': 'immediate',
                    'transition': {'duration': 300}
                }
            ],
            'label': changing_var,
            'method': 'animate'
        }
        sliders_dict['steps'].append(slider_step)

    figure['layout']['sliders'] = [sliders_dict]    

    if kwargs.get('plot'):
        if kwargs.get('filename'):
            plotly.offline.plot(figure, filename=kwargs['filename'])
        else:
            plotly.offline.iplot(figure)


def example_FT_properties():
    sigma = 10
    narray = 101
    
    fig = tools.make_subplots(rows=8, cols=2,
                              specs=[[{'is_3d': True}, {'is_3d': True}] for i in range(8)])
    ft = matrixDFT.MatrixFourierTransform()
    
    x, gaussian_array = make1DGaussian(narray, sigma, x0 = 10)
    gaussian_array = np.array([gaussian_array])
    ft_gaussian = ft.perform(gaussian_array, 9, (1, 81))
    
    x_data = np.linspace(-40, 40, 81)
    
    func1 = lambda x, x0, sigma: np.exp(-(x-x0)**2./(2.*sigma**2.))
    func2 = lambda x, x0, sigma: x * np.exp(-(x-x0)**2./(2.*sigma**2.))
    
    x_input = np.linspace(-50, 50, 101)
    
    y = func1(x_input, 0., sigma)
    trace = complex_3dplot(x_input, y, plot=False, return_traces=True)
    fig.append_trace(trace[0], 1, 1)
    fig.append_trace(trace[1], 1, 1)
    
    ft_y = ft.perform(np.array([y]), 9, (1, 81))
    trace = complex_3dplot(x_data, ft_y[0], plot=False, return_traces=True)
    fig.append_trace(trace[0], 1, 2)
    fig.append_trace(trace[1], 1, 2)
    
    y = func2(x_input, 0., sigma)
    trace = complex_3dplot(x, y, plot=False, return_traces=True)
    fig.append_trace(trace[0], 2, 1)
    fig.append_trace(trace[1], 2, 1)
    
    ft_y = ft.perform(np.array([y]), 9, (1, 81))
    trace = complex_3dplot(x_data, ft_y[0], plot=False, return_traces=True)
    fig.append_trace(trace[0], 2, 2)
    fig.append_trace(trace[1], 2, 2)
    
    
    y = func1(x_input, 0., sigma) * 1j
    trace = complex_3dplot(x_input, y, plot=False, return_traces=True)
    fig.append_trace(trace[0], 3, 1)
    fig.append_trace(trace[1], 3, 1)
    
    ft_y = ft.perform(np.array([y]), 9, (1, 81))
    trace = complex_3dplot(x_data, ft_y[0], plot=False, return_traces=True)
    fig.append_trace(trace[0], 3, 2)
    fig.append_trace(trace[1], 3, 2)
    
    y = func2(x_input, 0., sigma) * 1j
    trace = complex_3dplot(x, y, plot=False, return_traces=True)
    fig.append_trace(trace[0], 4, 1)
    fig.append_trace(trace[1], 4, 1)
    
    ft_y = ft.perform(np.array([y]), 9, (1, 81))
    trace = complex_3dplot(x_data, ft_y[0], plot=False, return_traces=True)
    fig.append_trace(trace[0], 4, 2)
    fig.append_trace(trace[1], 4, 2)
    
    
    
    y = func1(x_input, 0., sigma) * (1. + 1j)
    trace = complex_3dplot(x_input, y, plot=False, return_traces=True)
    fig.append_trace(trace[0], 5, 1)
    fig.append_trace(trace[1], 5, 1)
    
    ft_y = ft.perform(np.array([y]), 9, (1, 81))
    trace = complex_3dplot(x_data, ft_y[0], plot=False, return_traces=True)
    fig.append_trace(trace[0], 5, 2)
    fig.append_trace(trace[1], 5, 2)
    
    y = func2(x_input, 0., sigma) * (1. + 1j)
    trace = complex_3dplot(x, y, plot=False, return_traces=True)
    fig.append_trace(trace[0], 6, 1)
    fig.append_trace(trace[1], 6, 1)
    
    ft_y = ft.perform(np.array([y]), 9, (1, 81))
    trace = complex_3dplot(x_data, ft_y[0], plot=False, return_traces=True)
    fig.append_trace(trace[0], 6, 2)
    fig.append_trace(trace[1], 6, 2)
    
    
    y = func1(x_input, 10., sigma)
    trace = complex_3dplot(x_input, y, plot=False, return_traces=True)
    fig.append_trace(trace[0], 7, 1)
    fig.append_trace(trace[1], 7, 1)
    
    ft_y = ft.perform(np.array([y]), 9, (1, 81))
    trace = complex_3dplot(x_data, ft_y[0], plot=False, return_traces=True)
    fig.append_trace(trace[0], 7, 2)
    fig.append_trace(trace[1], 7, 2)
    
    y = func1(x_input, 10., sigma) * 1j
    trace = complex_3dplot(x, y, plot=False, return_traces=True)
    fig.append_trace(trace[0], 8, 1)
    fig.append_trace(trace[1], 8, 1)
    
    ft_y = ft.perform(np.array([y]), 9, (1, 81))
    trace = complex_3dplot(x_data, ft_y[0], plot=False, return_traces=True)
    fig.append_trace(trace[0], 8, 2)
    fig.append_trace(trace[1], 8, 2)
    
    
    fig['layout'].update(title='Properties of Fourier Transform',
                         height=1600, width=600)
    
    plotly.offline.iplot(fig)
