import matplotlib.pyplot as plt
import random
import time
import sys
import numpy as np
import networkx as nx
from helper_functions import graph_to_mat, mat_to_graph, gen_permutation_matrix, get_chr
from helper_functions import get_color_indice_array, density, floyd_warshall_networkx, loss
from helper_functions import get_color_adj_graph
from helper_functions import get_attraction_coefficient, get_attraction_coefficient_binary
# import keyboard
import readchar
# Add widgets
import matplotlib.widgets as widgets

def handle_close(evt):
    plt.close('all')
    exit()


def graph_painter_exp_attraction_visual(adj, num_iters = 3, phi_init = None, central_term_only = False, pause_duration = 0.01):
    n = adj.shape[0]  
    num_nodes = n                          
    # phi = uniform randomly between -1 and 1
    if phi_init is None:
        curr_phi = np.random.uniform(-1, 1, n)*np.pi*0
    else:
        curr_phi = phi_init
    min_colors = n
    best_coloring = [[i] for i in range(n)]
    assigned_colors = [i for i in range(n)]
    phi_history = []
    phi_history.append(curr_phi)
    total_iter_cnt = 0
    # attraction_coeff = get_attraction_coefficient(adj)
    attraction_coeff = get_attraction_coefficient_binary(adj)
    num_points_to_plot = 100
    # In phi_history, copy the curr_phi num_points_to_plot times
    for i in range(num_points_to_plot-1):
        phi_history.append(curr_phi)
    # Convert phi_history to numpy array
    phi_history = np.array(phi_history)
    X = np.linspace(1, num_points_to_plot, num_points_to_plot)
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('close_event', handle_close)

    curr_mid_points = [0 for i in range(n)]
    curr_mid_points = np.array(curr_mid_points)
    mid_points_history = []
    mid_points_history.append(curr_mid_points)
    for i in range(num_points_to_plot-1):
        mid_points_history.append(curr_mid_points)
    # Convert mid_points_history to numpy array
    mid_points_history = np.array(mid_points_history)
    
    # Create a line object for each node
    lines = []
    # print(len(X))
    # print(phi_history[:,3].shape)
    for i in range(n):
        line, = ax.plot(X, phi_history[:,i], label=f'Node {i}')
        lines.append(line)

    mid_point_dummy_lines = []
    markersize = 5
    if num_nodes > 30:
        markersize = 3
    if num_nodes > 50:
        markersize = 2
    if num_nodes > 100:
        markersize = 1
    for i in range(n):
        mid_point_dummy_line, = ax.plot(X, mid_points_history[:,i], color='black' , marker='.', markersize=markersize, linestyle='None', label=f'Mid Point {i}')
        mid_point_dummy_lines.append(mid_point_dummy_line)
    # Add sliders logarithmic sliders
    ax.set_ylim(-np.pi*1.02, np.pi*1.02)
    # ax.set_ylim(-4,4)
    x_start = 0
    x_end = num_points_to_plot
    ax.set_xlim(x_start, x_end)
    ax.set_xlabel('Iteration')


    # Add on-off widget
    axcolor = 'lightgoldenrodyellow'
    ax_on = plt.axes([0.6, 0.9, 0.15, 0.06], facecolor=axcolor)
    on_button = widgets.Button(ax_on, 'On/Off', color=axcolor, hovercolor='0.975')
    on_button.label = 'On'
    def on_button_clicked(event):
        if on_button.label == 'On':
            on_button.label = 'Off'
        else:
            on_button.label = 'On'
    on_button.on_clicked(on_button_clicked)

    # Add Terminate button
    ax_terminate = plt.axes([0.8, 0.9, 0.15, 0.06], facecolor=axcolor)
    terminate_button = widgets.Button(ax_terminate, 'Terminate', color=axcolor, hovercolor='0.975')
    terminate_button.label = 'On'
    def terminate_button_clicked(event):
        terminate_button.label = 'Off'
    terminate_button.on_clicked(terminate_button_clicked)

    # midpoint_scatter = ax.scatter([], [], color='black', s=20, label='Mid Point Phases')
    ax_step_size = plt.axes([0.1, 0.96, 0.15, 0.04], facecolor=axcolor)
    step_size_slider = widgets.Slider(ax_step_size, 'Step Size', valmin=-7, valmax=2,
                                       valinit=-5, valstep=0.1, color=axcolor)
    
    ax_sigma_exp = plt.axes([0.1, 0.93, 0.15, 0.04], facecolor=axcolor)
    sigma_exp_slider = widgets.Slider(ax_sigma_exp, 'Sigma Exp', valmin=-7, valmax=2,
                                       valinit=0, valstep=0.1, color=axcolor)
    
    ax_sigma_noise = plt.axes([0.1, 0.90, 0.15, 0.04], facecolor=axcolor)
    sigma_noise_slider = widgets.Slider(ax_sigma_noise, 'Sigma Noise', valmin=-7, valmax=2,
                                         valinit=-5, valstep=0.1, color=axcolor)
    
    ax_sigma_attraction = plt.axes([0.4, 0.96, 0.15, 0.04], facecolor=axcolor)
    sigma_attraction_slider = widgets.Slider(ax_sigma_attraction, 'Sigma Attraction', valmin=-7, valmax=2,
                                              valinit=0, valstep=0.1, color=axcolor)
   
    ax_attraction_step_size = plt.axes([0.4, 0.93, 0.15, 0.03], facecolor=axcolor)
    attraction_step_size_slider = widgets.Slider(ax_attraction_step_size, 'Attraction Step Size', valmin=-7, valmax=2,
                                                  valinit=-5, valstep=0.1, color=axcolor)

    ax_tep_size_text_box = plt.axes([0.25, 0.965, 0.05, 0.03], facecolor=axcolor)
    step_size_text_box = widgets.TextBox(ax_tep_size_text_box, label = '', initial=str(10**step_size_slider.val))
    
    ax_sigma_exp_text_box = plt.axes([0.25, 0.935, 0.05, 0.03], facecolor=axcolor)
    sigma_exp_text_box = widgets.TextBox(ax_sigma_exp_text_box, label = '', initial=str(10**sigma_exp_slider.val))

    ax_sigma_noise_text_box = plt.axes([0.25, 0.905, 0.05, 0.03], facecolor=axcolor)
    sigma_noise_text_box = widgets.TextBox(ax_sigma_noise_text_box, label = '', initial=str(10**sigma_noise_slider.val))

    ax_sigma_attraction_text_box = plt.axes([0.55, 0.965, 0.05, 0.03], facecolor=axcolor)
    sigma_attraction_text_box = widgets.TextBox(ax_sigma_attraction_text_box, label = '', initial=str(10**sigma_attraction_slider.val))
    
    ax_attraction_step_size_text_box = plt.axes([0.55, 0.930, 0.05, 0.03], facecolor=axcolor)
    attraction_step_size_text_box = widgets.TextBox(ax_attraction_step_size_text_box, label = '', initial=str(10**attraction_step_size_slider.val))
    
    ax_current_num_color_textbox = plt.axes([0.40, 0.905, 0.05, 0.03], facecolor=axcolor)
    current_num_color_textbox = widgets.TextBox(ax_current_num_color_textbox, label = 'Curr num color', initial=str(min_colors))

    ax_least_num_color_textbox = plt.axes([0.40, 0.875, 0.05, 0.03], facecolor=axcolor)
    least_num_color_textbox = widgets.TextBox(ax_least_num_color_textbox, label = 'Least num color', initial=str(min_colors))
    
    

    for i in range(num_iters):
        if terminate_button.label == 'Off':
            print("Terminating...")
            exit()
        while on_button.label == 'Off':
            # print("Waiting for button to be clicked...")
            plt.pause(0.1)
            if terminate_button.label == 'Off':
                print("Terminating...")
                exit()
        # read the values from the sliders
        step_size = 10**step_size_slider.val
        sigma_exp = 10**sigma_exp_slider.val
        sigma_noise = 10**sigma_noise_slider.val
        sigma_attraction = 10**sigma_attraction_slider.val
        attraction_step_size = 10**attraction_step_size_slider.val
        # Update the text boxes with the current values in scientific notation
        step_size_text_box.set_val(f'{step_size:.2e}')
        sigma_exp_text_box.set_val(f'{sigma_exp:.2e}')
        sigma_noise_text_box.set_val(f'{sigma_noise:.2e}')
        sigma_attraction_text_box.set_val(f'{sigma_attraction:.2e}')
        attraction_step_size_text_box.set_val(f'{attraction_step_size:.2e}')  
        total_iter_cnt += 1
        curr_phi = curr_phi + sigma_noise*np.random.randn(n)
        # compute the loss derivative
        phi_curr = np.reshape(curr_phi, (curr_phi.shape[0],))
        x = phi_curr[:, None] - phi_curr

        term1 = -np.sign(x) * np.exp(-np.abs(x) / sigma_exp)/sigma_exp
        term1[x == 0] = 0
        np.fill_diagonal(term1, 0) # to avoid self-derivative
        term2 = np.exp((x - 2 * np.pi) / sigma_exp)/sigma_exp # Sign not needed as x < 2pi                                         
        np.fill_diagonal(term2, 0)
        term3 = -np.exp(-(x + 2 * np.pi) / sigma_exp)/sigma_exp # Sign not needed as x < 2pi
        np.fill_diagonal(term3, 0)

        term1_attract = -np.sign(x) * np.exp(-np.abs(x) / sigma_attraction)/sigma_attraction
        term1_attract[x == 0] = 0
        np.fill_diagonal(term1_attract, 0) # to avoid self-derivative
        term2_attract = np.exp((x - 2 * np.pi) / sigma_attraction)/sigma_attraction # Sign not needed as x < 2pi
        np.fill_diagonal(term2_attract, 0)
        term3_attract = -np.exp(-(x + 2 * np.pi) / sigma_attraction)/sigma_attraction # Sign not needed as x < 2pi
        np.fill_diagonal(term3_attract, 0)


        loss_derivative = term1 + term2 + term3
        loss_derivative_clip = np.clip(loss_derivative, -10, 10)

        attraction_derivative = term1_attract + term2_attract + term3_attract
        attraction_derivative_clip = np.clip(attraction_derivative, -10, 10)
        # The clipping is important. Because of noise, two connected nodes can come close and
        # blow up the derivative.
        curr_phi = curr_phi - step_size* np.sum(adj*loss_derivative_clip, axis=1) + attraction_step_size * np.sum(attraction_coeff*attraction_derivative_clip, axis=1)
        curr_phi[curr_phi > np.pi] = curr_phi[curr_phi > np.pi] - 2*np.pi
        curr_phi[curr_phi < -np.pi] = curr_phi[curr_phi < -np.pi] + 2*np.pi
        # add the current phi to the history, phi_history is a numpy array of shape (num_iters, n)
        phi_history = np.append(phi_history, [curr_phi], axis=0)

        # update the plot
        for j in range(n):
            lines[j].set_ydata(phi_history[-num_points_to_plot:, j])
        # update the x data
        X = np.linspace(x_start, x_end, num_points_to_plot)
        for j in range(n):
            lines[j].set_xdata(X)
        
        # for j in range(len(mid_point_dummy_lines)):
        #     mid_point_dummy_lines[j].set_xdata(X)

        # update the plot
        ax.set_xlim(x_start, x_end)
        x_start += 1
        x_end += 1
        
        P, order = gen_permutation_matrix(curr_phi)
        num_chr, color_blocks = get_chr(P, adj, order)
        curr_color = 0
        for blocks in color_blocks:
            for nodes in blocks:
                assigned_colors[nodes] = curr_color
            curr_color += 1
        phase_ordering = np.argsort(curr_phi)
        new_midpoints = np.ones(n) * (-1.2) * np.pi
        for j in range(n-1):
            if assigned_colors[phase_ordering[j]] != assigned_colors[phase_ordering[j+1]]:
                new_midpoints[phase_ordering[j]] = (curr_phi[phase_ordering[j]] + curr_phi[phase_ordering[j+1]]) / 2
        if assigned_colors[phase_ordering[-1]] != assigned_colors[phase_ordering[0]]:
            new_midpoints[phase_ordering[-1]] = np.pi
        curr_mid_points = new_midpoints
        mid_points_history = np.append(mid_points_history, [curr_mid_points], axis=0)
        # update the mid point lines
        for j in range(n):
            mid_point_dummy_lines[j].set_ydata(mid_points_history[-num_points_to_plot:, j])
            mid_point_dummy_lines[j].set_xdata(X)
        # For the points in the mid_point_phase_list, plot a scatter plot using the phase
        # on y axis and the current x value on x axis in the same plot
        # mid_point_phase_list = np.array(mid_point_phase_list)
        
        
        # for k in range(len(mid_point_phase_list)):
        #     mid_point_dummy_lines[k].set_ydata(np.full(X.shape, mid_points_history[-1, k]))
        
        # for k in range()
        # ax.plot([x_end for i in range (len(mid_point_phase_list))], mid_point_phase_list, color='black', s=20, label='Mid Point Phases')
        # x_values = np.full(len(mid_point_phase_list), x_end)
        # midpoint_scatter.set_offsets(np.column_stack([x_values, mid_point_phase_list]))
        # midpoint_line[0].set_xdata(x_values)
        # midpoint_line[0].set_ydata(mid_point_phase_list)
        
        plt.draw()
        plt.pause(pause_duration)
        # Ordered color_blocks according to 
        # Update the text box with the current number of colors
        current_num_color_textbox.set_val(str(num_chr))
        if num_chr < min_colors:
            # print(total_iter_cnt, " ", num_chr)
            min_colors = num_chr
            best_coloring = color_blocks
            least_num_color_textbox.set_val(str(min_colors))
            # plot a yline at current iteration
            ax.axvline(x=i, color='r', ymin = -4, ymax = 4, linewidth = 3)
        # convert phi_history to numpy array
    plt.pause(2)
    
    phi_history = np.array(phi_history)
    return min_colors, best_coloring, phi_history


if __name__ == "__main__":
    graph_name = 'dsjc250.5'
    # graph_name = 'flat300_28_0'
    # Call the function to animate the sine wave
    # animate_sine_wave2()
    num_iters=5000
    G = mat_to_graph('test_graphs/'+graph_name+'.mat')
    adj = nx.adjacency_matrix(G).toarray()
    adj = np.asarray(adj, dtype=np.int32)
    density_original = density(adj)
    num_colors, best_coloring, phi_history = graph_painter_exp_attraction_visual(adj, num_iters,
                                            central_term_only=False, pause_duration = 0.0001)
    print("num_colors: ", num_colors)
    
