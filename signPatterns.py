import numpy as np

def generate_sign_patterns():
    def create_circle_sign(points=50, radius=1.0, noise=0.1):
        t = np.linspace(0, 2*np.pi, points)
        x = radius * np.cos(t) + np.random.normal(0, noise, points)
        y = radius * np.sin(t) + np.random.normal(0, noise, points)
        z = np.zeros(points) + np.random.normal(0, noise/2, points)
        return np.column_stack((x, y, z))

    def create_wave_sign(points=50, noise=0.1):
        t = np.linspace(0, 4*np.pi, points)
        x = t/4 + np.random.normal(0, noise, points)
        y = np.sin(t) + np.random.normal(0, noise, points)
        z = np.cos(t/2) + np.random.normal(0, noise, points)
        return np.column_stack((x, y, z))

    def create_zigzag_sign(points=50, noise=0.1):
        t = np.linspace(0, 4, points)
        x = t + np.random.normal(0, noise, points)
        y = np.abs((t % 1) - 0.5) + np.random.normal(0, noise, points)
        z = np.zeros(points) + np.random.normal(0, noise/2, points)
        return np.column_stack((x, y, z))
    
    signs_database = {
        'circle': {
            'reference': create_circle_sign(noise=0.05),
            'variations': [
                create_circle_sign(noise=0.1),
                create_circle_sign(noise=0.15),
                create_circle_sign(radius=1.2, noise=0.1)
            ]
        },
        'wave': {
            'reference': create_wave_sign(noise=0.05),
            'variations': [
                create_wave_sign(noise=0.1),
                create_wave_sign(noise=0.15),
                create_wave_sign(noise=0.2)
            ]
        },
        'zigzag': {
            'reference': create_zigzag_sign(noise=0.05),
            'variations': [
                create_zigzag_sign(noise=0.1),
                create_zigzag_sign(noise=0.15),
                create_zigzag_sign(noise=0.2)
            ]
        }
    }
    
    return signs_database