'''
    Module with a library of functions that define pulse shapes. 
    Each of them returns a shape function that gives the pulse value(s)
    when applied on the desired value(s) of time.
'''

import numpy as np

def square(t_start=0, t_end=1, amp=1, offset=0):
    '''
        Creates a square pulse of amplitude amp and that continues
        from t_start until t_end. The offset value before and after the pulse 
        can be also specified.
        Convenient for defining pulses in terms of 
        normalized time tau = (t-t_start) / (t_end - t_start), as 0<=tau<=1, 
        which corresponds to the default values.

        Keyword Arguments
        -----------------
        t_start: float
            Time point at which the pulse begins. Default is 0.
        t_end: float
            Time point at which the pulse ends. Default is 1. 
        amp: float
            Amplitude of the pulse. Default is 1.
        offset: float
            Default value of a control variable before and after the pulse.
            Default is 0. 

        Returns
        -------
        square_pulse: function
            Function that defines a square pulse when applied to the desired
            time values
            
        '''
    def square_pulse(t):    
        # making the time input a float array
        try:
            t = np.array(t, dtype=float)
        except ValueError:
            print("Unable to convert {type} object to float array".format(type = type(t)))

        # creating a pulse with offset values at all times first
        pulse_arr = np.full(t.shape, offset, dtype=float)

        # selecting the time points that lie within the pulse
        pulse_mask = np.greater_equal(t, t_start) & np.less_equal(t, t_end)
        pulse_arr[pulse_mask] = amp
        return pulse_arr

    return square_pulse

def triangle(t_start=0, t_end=1, amp=1, offset=0):
    '''
        Creates a symmetric triangular pulse of amplitude amp and that continues
        from t_start until t_end. The pulse starts and ends at a constant value
        of offset.
        Convenient for defining pulses in terms of 
        normalized time tau = (t-t_start) / (t_end - t_start), as 0<=tau<=1, 
        which corresponds to the default values.

        Keyword Arguments
        -----------------
        t_start: float
            Time point at which the pulse begins. Default is 0.
        t_end: float
            Time point at which the pulse ends. Default is 1. 
        amp: float
            Amplitude of the pulse. Default is 1.
        offset: float
            Default value of a control variable before and after the pulse.
            Default is 0. 

        Returns
        -------
        square_pulse: function
            Function that defines a triangular pulse when applied to the desired
            time values.
 
        '''
    def triangular_pulse(t):    
        # making the time input a float array
        try:
            t = np.array(t, dtype=float)
        except ValueError:
            print("Unable to convert {type} object to float array".format(type = type(t)))

        # creating a pulse with offset values at all times first
        pulse_arr = np.full(t.shape, offset, dtype=float)

        # selecting the time points that lie within the rising and falling edges
        t_center = (t_start + t_end) / 2
        rising_edge_mask = np.greater_equal(t, t_start) & np.less_equal(t, t_center)
        falling_edge_mask = np.greater(t, t_center) & np.less_equal(t, t_end)

        t_rising = t[rising_edge_mask]
        t_falling = t[falling_edge_mask]
        pulse_arr[rising_edge_mask] = offset + ((amp - offset) * 
                                (t_rising - t_start) / (t_center - t_start))

        pulse_arr[falling_edge_mask] = offset + ((amp - offset) *  
                               (t_end - t_falling) / (t_end - t_center))

        return pulse_arr

    return triangular_pulse


def shifted_gauss(t_start=0, t_end=1, amp=1, offset=1, sigma=0.1 ):
    '''
        Creates a Gaussian pulse of amplitude amp and relative width
        sigma that lasts from t_start until t_end. The pulse is shifted and 
        scaled vertically so that the pulse values at the beginning and 
        at the end are equal to offset. 

        Convenient for defining pulses in terms of 
        normalized time tau = (t-t_start) / (t_end - t_start), as 0<=tau<=1, 
        which corresponds to the default values.

        Keyword Arguments
        -----------------
        t_start: float
            Time point at which the pulse begins. Default is 0.
        t_send: float
            Time point at which the pulse ends. Default is 1. 
        amp: float
            Amplitude of the pulse. Default is 1. 
        offset: float
            The shift of the Gaussian function vertically; corresponds to the
            value of the pulse at t_start and t_end. Assumed to be constant
            before and after the pulse. Default is 0.
        sigma: float
            Half-width of the pulse (in units of time). Default is 0.1.

        Returns
        -------
        shifted_gauss_pulse: function
            Function that defines a shifted Gaussian pulse when applied 
            to the desired time value(s)
        '''
    def shifted_gauss_pulse(t):

        # making the time input a float array
        if isinstance(t, (int,float,list,tuple,range)):
            t = np.array(t, dtype=float)
        # switching to dimensionless units for convenience
        tau = (t - t_start) / (t_end - t_start)
        sig = sigma / (t_end - t_start)

        # creating a pulse with offset values at all times first
        pulse_arr = np.full(t.shape, offset, dtype=float)
        # selecting the time points that lie within the pulse
        pulse_mask = np.greater_equal(t, t_start) & np.less_equal(t, t_end)
        tau_masked = tau[pulse_mask]

        # calculating and setting the correct values during the pulse
        num = (np.exp( -(tau_masked-0.5)**2 /(2 * sig ** 2))
                        - np.exp(-1 / (8 * sig ** 2)))
        denom = 1 - np.exp(-1 /(8 * sig ** 2))

        pulse_arr[pulse_mask] = offset + (amp - offset) * num / denom

        return pulse_arr
    return shifted_gauss_pulse

if __name__=='__main__':
    print('Creating a Gaussian pulse:')
    gauss_pulse = shifted_gauss(t_start=0, t_end=10, amp=5, sigma=2, offset=1)
    times = np.arange(20)
    print('Pulse: ', gauss_pulse(times))
    # import matplotlib.pyplot as plt
    # plt.plot(times, gauss_pulse(times))
    

    