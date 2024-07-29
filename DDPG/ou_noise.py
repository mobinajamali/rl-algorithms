# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
import numpy as np

class OUActionNoise():
    '''
    Implement the Ornstein-Uhlenbeck process to generate noise for exploration 
    in continuous action spaces
    '''
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        '''
        mu = mean for the noise
        sigma = std
        theta = mean reversion rate
        dt = process time step
        x0 = starting value
        '''
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        # reset the noise
        self.reset()

    def __call__(self):
        '''
        Generate the next noise value based on the Ornstein-Uhlenbeck process
        '''
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
    
    def reset(self):
        '''
        Reset the noise process to the initial value
        '''
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)




