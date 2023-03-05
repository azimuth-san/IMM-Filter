### Constant Velocity Model

$$
\boldsymbol{x}_{t+1} =
\begin{bmatrix}
1 & T & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & T \\
0 & 0 & 0 & 1 \\
\end{bmatrix} \boldsymbol{x}_{t}
+\begin{bmatrix}
\frac{1}{2}T^2 & 0 \\
T & 0 \\
0 & \frac{1}{2}T^2 \\
0 & T \\
\end{bmatrix}
\boldsymbol{w}_t
$$

$$
\boldsymbol{z}_{t} =
\begin{bmatrix}
1 & 0 & 0 & 0  \\
0 & 0 & 1 & 0  \\
\end{bmatrix} \boldsymbol{x}_t + \boldsymbol{v}_t
$$

$$
\boldsymbol{x}_{t} = 
\begin{bmatrix}
x_t & \dot{x_{t}} & y_t &\dot{y_{t}}
\end{bmatrix},~~ 
\boldsymbol{z}_{t} = 
\begin{bmatrix}
x_t & y_t 
\end{bmatrix},~~ 
$$

$$
\boldsymbol{w}_{t} \sim N(\boldsymbol{0}, Q),~~
\boldsymbol{v}_{t} \sim N(\boldsymbol{0}, R).~~
$$

<br>
### Coordinated Turn Model
$$
\boldsymbol{x}_{t+1} =
\begin{bmatrix}
1 & {\rm{sin}} \omega_t T / \omega_t  & 0 & -(1-{\rm{cos}} \omega_t T) / \omega_t & 0 \\
0 & {\rm{cos}} \omega_t T & 0 & -{\rm{sin}} \omega_t T & 0 \\
0 & (1-{\rm{cos}} \omega_t T) / \omega_t & 1 &  {\rm{sin}} \omega_t T / \omega_t   & 0 \\
0 & {\rm{sin}} \omega_t T & 0 & {\rm{cos}} \omega_t T & 0 \\
0 & 0 & 0 & 0 & 1 \\
\end{bmatrix} \boldsymbol{x}_t
+\begin{bmatrix}
\frac{1}{2}T^2 & 0 & 0 \\
T & 0 & 0 \\
0 & \frac{1}{2}T^2 & 0 \\
0 & T & 0 \\
0 & 0 & T \\
\end{bmatrix}
\boldsymbol{w}_t
$$

$$
\boldsymbol{z}_{t} =
\begin{bmatrix}
1 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 \\
\end{bmatrix}\boldsymbol{x}_t + \boldsymbol{v}_t
$$

$$
\boldsymbol{x}_{t} = 
\begin{bmatrix}
x_t & \dot{x_{t}} &y_t &\dot{y_{t}} & \omega_t
\end{bmatrix},~~
\boldsymbol{z}_{t} = 
\begin{bmatrix}
x_t & y_t 
\end{bmatrix},~~ 
$$

$$
\boldsymbol{w}_{t} \sim N(\boldsymbol{0}, Q),~~
\boldsymbol{v}_{t} \sim N(\boldsymbol{0}, R).~~
$$

