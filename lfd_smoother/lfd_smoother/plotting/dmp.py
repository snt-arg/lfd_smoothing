
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.default'] = 'regular'

class DMPAnalysis:

    def __init__(self) -> None:
        pass

    def plot_3d(self, original_dmp, smooth_dmp):
        # Make a figure for 3D trajectory
        fig = plt.figure(figsize=(4, 4))
        
        # Create 3D axis
        ax_traj = fig.add_subplot(111, projection='3d')
        
        # Plot original DMP
        ax_traj.plot(original_dmp.positions[:, 0], original_dmp.positions[:, 1], original_dmp.positions[:, 2], label='$p_{o,dmp}$', linestyle='--', color='royalblue')
        
        # Plot smooth DMP
        ax_traj.plot(smooth_dmp.positions[:, 0], smooth_dmp.positions[:, 1], smooth_dmp.positions[:, 2], label='$p_{f,dmp}$', linestyle='-', color='firebrick')

        ax_traj.set_xlabel('x [m]')
        ax_traj.set_ylabel('y [m]')
        ax_traj.set_zlabel('z [m]')
        
        # Add legend to differentiate the original and smooth DMP
        ax_traj.legend(fontsize=12)
        
        # plt.tight_layout()
        plt.show() 

    def calc_abs_jerk(self, traj):
        # Calculate jerk for original_dmp
        velocities = np.array(traj.velocities)
        times = np.array(traj.ts)
        accelerations = np.gradient(velocities, times, edge_order=1)
        jerks = np.gradient(accelerations, times, edge_order=1)
        abs_jerks = np.abs(jerks)
        return abs_jerks

    def plot_abs_jerk(self,original_dmp, smooth_dmp, scaled_dmp):
        # Create plot for jerk comparison
        plt.figure(figsize=(5, 2))
        plt.plot(original_dmp.ts, self.calc_abs_jerk(original_dmp), label='Original DMP', linestyle='--', color='royalblue')
        plt.plot(scaled_dmp.ts, self.calc_abs_jerk(scaled_dmp), label='Original DMP - Scaled', linestyle='-.', color='turquoise')
        plt.plot(smooth_dmp.ts, self.calc_abs_jerk(smooth_dmp), label='Smooth DMP', linestyle='-', color='firebrick')
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
        # plt.title('Absolute Jerk Comparison')
        plt.xlabel('Normalized time s')
        plt.ylabel('|Jerk|')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_with_kin_lims(self,scaled_dmp):
        ts = scaled_dmp.ts
        data = {
            "vel [rad/s]": scaled_dmp.yds,
            "accl [rad/s${}^2$]": scaled_dmp.ydds,
            "jerk [rad/s${}^3$]": scaled_dmp.yddds
        }
        
        vel_limits = [2, 1.1, 1.5, 1.25, 3, 1.5, 3]
        acc_limits = [10, 10, 10, 10, 10, 10, 10]
        jerk_limits = [5000, 5000, 5000, 5000, 5000, 5000, 5000]
        
        limits = {
            "vel [rad/s]": vel_limits,
            "accl [rad/s${}^2$]": acc_limits,
            "jerk [rad/s${}^3$]": jerk_limits
        }
        
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        
        plt.figure(figsize=(8, 2.5))
        
        for idx, (key, values) in enumerate(data.items()):
            plt.subplot(1, 3, idx + 1)
            
            for i, dof in enumerate(range(7)):  # Assuming 7 DOF
                
                color = colors[i]
                limit = limits[key][i]
                
                plt.plot(ts, values[:, i], label=f'Joint {i+1}', linestyle="-", linewidth=2, color="royalblue", alpha=0.5)
                plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

                # Highlight Violations
                above_limit = np.where(values[:, i] > limit)
                plt.scatter(ts[above_limit], values[above_limit, i], color=color, marker='x' , zorder=3)
                
            
            plt.xlabel('Time [s]')
            plt.ylabel(key, labelpad=-5)
            
        plt.tight_layout()
        plt.show()


    def plot_vel_acc_kin_lims(self, scaled_dmp, smooth_dmp):
        ts_scaled = scaled_dmp.ts
        ts_smooth = smooth_dmp.ts

        vel_limits = [2, 1.1, 1.5, 1.25, 3, 1.5, 3]
        acc_limits = [10, 10, 10, 10, 10, 10, 10]
        
        limits = {
            "vel [rad/s]": vel_limits,
            "accl [rad/s${}^2$]": acc_limits
        }
        
        colors = ['r', 'r', 'r', 'r', 'r', 'r', 'r']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        
        for idx, name in enumerate(['vel [rad/s]', 'accl [rad/s${}^2$]']):
            for plot_row, (dmp, ts) in enumerate([(scaled_dmp, ts_scaled), (smooth_dmp, ts_smooth)]):
                ax = axes[plot_row, idx]
                
                if name == 'vel [rad/s]':
                    values = dmp.yds
                else:  # 'accl [rad/s${}^2$]'
                    values = dmp.ydds

                for i, dof in enumerate(range(7)):  # Assuming 7 DOF
                    color = colors[i]
                    limit = limits[name][i]
                    
                    # Highlight Violations
                    above_limit = np.where(values[:, i] > limit)
                    ax.scatter(ts[above_limit], values[above_limit, i], label="violation", color=color, marker='o' , s=30,  zorder=3, alpha=0.7)
                    
                    # plot_color = colors[i] if above_limit[0].size > 0 else 'royalblue'
                    plot_color = 'royalblue'

                    ax.plot(ts, values[:, i], linestyle="-", label="trajectory",  linewidth=2, color=plot_color, alpha=0.7)
                    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
                    ax.grid(True, linestyle='--', alpha=0.6)

                    

        axes[0, 0].set_ylabel('$\dot{q}_{s,dmp}$ [rad/s]' , labelpad=-5, fontsize=26)
        axes[0, 1].set_ylabel('$\ddot{q}_{s,dmp}$ [rad/s${}^2$]', labelpad=-3, fontsize=26)
        axes[1, 0].set_ylabel('$\dot{q}_{f,dmp}$ [rad/s]', labelpad=-5, fontsize=26)
        axes[1, 1].set_ylabel('$\ddot{q}_{f,dmp}$ [rad/s${}^2$]', labelpad=-5, fontsize=26)

        axes[1, 0].set_xlabel('Time [s]', fontsize=22)
        axes[1, 1].set_xlabel('Time [s]', fontsize=22)

        handles, labels = axes[0,1].get_legend_handles_labels()

        # Use a dictionary to remove duplicates
        unique = {label: handle for handle, label in zip(handles, labels)}

        plt.legend(unique.values(), unique.keys(), loc='upper right', fontsize=16, markerscale=1)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig_filename = f'/tmp/dmp_kin.pdf'
        plt.savefig(fig_filename)
        plt.close()  # Close the figure to free up memory


    def plot_compare_jerks(self, scaled_dmp, smooth_dmp):
        ts_scaled = scaled_dmp.ts
        ts_smooth = smooth_dmp.ts

        data = {
            "Scaled DMP $\dddot{q}$ [rad/s${}^3$]": scaled_dmp.yddds,
            "Smooth DMP $\dddot{q}$ [rad/s${}^3$]": smooth_dmp.yddds
        }

        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        ax2 = ax1.twinx()

        color1 = "steelblue"
        color2 = "darkred"

        for label, values in data.items():
            for dof in range(values.shape[1]):
                if label.startswith('Scaled'):
                    ax1.plot(ts_scaled, values[:, dof], color=color1, linestyle="-", linewidth=2, alpha=0.5)
                    ax1.tick_params(axis='y', labelcolor=color1)
                    # for picknplace
                    # ax1.set_ylim(-480, 480)
                    ax1.set_ylim(-800, 800)

                else:  # Smooth
                    ax2.plot(ts_smooth, values[:, dof], color=color2, linewidth=2, alpha=0.7)
                    ax2.tick_params(axis='y', labelcolor=color2)
                    ax2.set_ylim(-300, 300)


        y_max_scaled = np.max(scaled_dmp.yddds)
        y_min_scaled = np.min(scaled_dmp.yddds)
        y_max_smooth = np.max(smooth_dmp.yddds)
        y_min_smooth = np.min(smooth_dmp.yddds)

        # Annotations for max and min values
        ax1.axhline(y=y_max_scaled, color=color1, linestyle='--') 
        # ax1.text(ts_scaled[int(len(ts_scaled)*0.8)], y_max_scaled - 55, f'Max: {y_max_scaled:.2f}', color="black", verticalalignment='bottom', bbox=dict(boxstyle="round", fc=color1, alpha=0.5), fontsize=14)
        ax1.text(ts_scaled[int(len(ts_scaled)*0.8)], y_max_scaled - 100, f'Max: {y_max_scaled:.2f}', color="black", verticalalignment='bottom', bbox=dict(boxstyle="round", fc=color1, alpha=0.5), fontsize=14)
        ax1.axhline(y=y_min_scaled, color=color1, linestyle='--') 
        # ax1.text(ts_scaled[int(len(ts_scaled)*0.8)], y_min_scaled + 55, f'Min: {y_min_scaled:.2f}', color="black", verticalalignment='top', bbox=dict(boxstyle="round", fc=color1, alpha=0.5), fontsize=14)
        ax1.text(ts_scaled[int(len(ts_scaled)*0.8)], y_min_scaled -40, f'Min: {y_min_scaled:.2f}', color="black", verticalalignment='top', bbox=dict(boxstyle="round", fc=color1, alpha=0.5), fontsize=14)

        ax2.axhline(y=y_max_smooth, color=color2, linestyle='--') 
        ax2.text(ts_smooth[int(len(ts_smooth)*0.8)], y_max_smooth + 15, f'Max: {y_max_smooth:.2f}', color="black", verticalalignment='bottom', bbox=dict(boxstyle="round", fc=color2, alpha=0.5), fontsize=14)
        ax2.axhline(y=y_min_smooth, color=color2, linestyle='--') 
        ax2.text(ts_smooth[int(len(ts_smooth)*0.8)], y_min_smooth - 15, f'Min: {y_min_smooth:.2f}', color="black", verticalalignment='top', bbox=dict(boxstyle="round", fc=color2, alpha=0.5), fontsize=14)

        ax1.set_ylabel("$\dddot{q}_{s,dmp}$ [rad/s${}^3$]", labelpad=10, color=color1, fontsize=24)
        ax2.set_ylabel("$\dddot{q}_{f,dmp}$ [rad/s${}^3$]", labelpad=10, color=color2, fontsize=24)
        ax1.set_xlabel('Time [s]', labelpad=10, fontsize=20)

        # Add gridlines for better readability
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax2.grid(False)  # to avoid overlapping grids

        # Add a legend to clarify which plot corresponds to which data set
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right')

        # plt.title('Comparison of Scaled and Smooth DMP Jerks', fontsize=16, pad=15)
        plt.tight_layout()
        fig_filename = f'/tmp/dmp_jerk.pdf'
        plt.savefig(fig_filename)
        plt.close()  # Close the figure to free up memory