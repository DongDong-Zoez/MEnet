# MEnet
A mask ensemble network architecture

### Pursuit

1. The generated mask may contain noise
2. The boundary of generated mask may be rugged

### Idea

1. Use multiple encoders to determine the best mask generator
2. The important pixels should be proposed by numerous models
3. Apply confident score to determine the importance of one single pixels
4. The output of each encoder should be a mask

### Architecture

![MEnet](https://github.com/DongDong-Zoez/MEnet/blob/032ed91370759446f30d1a1a3395a427bc64965c/assest/architecture.png)

### Loss function

$$
\begin{align}
\text{Encoder Loss}:&\ \frac{1}{k}\sum_{i=1}^k D_1(E_i(x)) \\
\text{Head Loss}:&\ D_2(H(x)) \\
\text{Ensemble Loss}:&\ \frac{1}{2}[\frac{1}{k}\sum_{i=1}^k D_1(E_i(x))] + \frac{1}{2} D_2(H(x))
\end{align}
$$


