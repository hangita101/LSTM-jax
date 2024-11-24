import jax
import jax.numpy as jnp
from jax import value_and_grad, grad
import optax

class LSTM():
    def __init__(self, input_size: int, hidden_size: int, rng):
        """
        Constructor for the LSTM class.
        -------------------------------
        Parameters:
        input_size (int): The number of input features.
        hidden_size (int): The number of units in the hidden state.
        rng (jax.random.PRNGKey): Random number generator key.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Properly handle random key splitting
        keys = jax.random.split(rng, 13)  # Split into 13 keys for all parameters
        
        # Initialize weights and biases for input gate
        self.Wii = jax.random.normal(keys[0], (input_size, hidden_size)) * 0.01
        self.Whi = jax.random.normal(keys[1], (hidden_size, hidden_size)) * 0.01
        self.bi = jax.random.normal(keys[2], (hidden_size,)) * 0.01
        
        # Initialize weights and biases for forget gate
        self.Wif = jax.random.normal(keys[3], (input_size, hidden_size)) * 0.01
        self.Whf = jax.random.normal(keys[4], (hidden_size, hidden_size)) * 0.01
        self.bf = jax.random.normal(keys[5], (hidden_size,)) * 0.01
        
        # Initialize weights and biases for cell gate
        self.Wig = jax.random.normal(keys[6], (input_size, hidden_size)) * 0.01
        self.Whg = jax.random.normal(keys[7], (hidden_size, hidden_size)) * 0.01
        self.bg = jax.random.normal(keys[8], (hidden_size,)) * 0.01
        
        # Initialize weights and biases for output gate
        self.Wio = jax.random.normal(keys[9], (input_size, hidden_size)) * 0.01
        self.Who = jax.random.normal(keys[10], (hidden_size, hidden_size)) * 0.01
        self.bo = jax.random.normal(keys[11], (hidden_size,)) * 0.01
        
        # Initialize default states (though these should typically be passed in)
        self.h_0 = jnp.zeros((hidden_size,))
        self.c_0 = jnp.zeros((hidden_size,))
    
    def get_initial_states(self):
        """Return the initial states."""
        return self.h_0, self.c_0
    
    @staticmethod
    def forward(params, X_t, h_t, c_t):
        """
        Forward pass of LSTM for a single time step.
        All state is passed in explicitly for better compatibility with JAX.
        """
        i = jax.nn.sigmoid(params['Wii'] @ X_t + params['Whi'] @ h_t + params['bi'])
        f = jax.nn.sigmoid(params['Wif'] @ X_t + params['Whf'] @ h_t + params['bf'])
        g = jax.nn.tanh(params['Wig'] @ X_t + params['Whg'] @ h_t + params['bg'])
        o = jax.nn.sigmoid(params['Wio'] @ X_t + params['Who'] @ h_t + params['bo'])
        
        # Update cell state and hidden state
        c_t_new = f * c_t + i * g
        h_t_new = o * jax.nn.tanh(c_t_new)  

        return c_t_new, h_t_new
    
    def params(self):
        """Return the parameters as a dictionary."""
        return {
            'Wii': self.Wii, 'Whi': self.Whi, 'bi': self.bi,
            'Wif': self.Wif, 'Whf': self.Whf, 'bf': self.bf,
            'Wig': self.Wig, 'Whg': self.Whg, 'bg': self.bg,
            'Wio': self.Wio, 'Who': self.Who, 'bo': self.bo,
        }
    
    @staticmethod
    def totalLoss(params, x, y):
        """
        Vectorized total loss function over all time steps using vmap.
        
        Parameters:
        params: dict containing LSTM parameters
        x: input sequence of shape (time_steps, input_size)
        y: target sequence of shape (time_steps, hidden_size)
        
        Returns:
        total loss across all time steps
        """
        # Create a function that processes a single time step
        def single_step(carry, inputs):
            h_t, c_t = carry
            x_t, y_t = inputs
            
            # Forward pass for single time step
            c_t_new, h_t_new = LSTM.forward(params, x_t, h_t, c_t)  # Changed from forwardPass to forward
            
            # Compute loss for this time step
            loss = jnp.sum((h_t_new - y_t)**2)
            
            return (h_t_new, c_t_new), loss

        # Initialize carrying states
        init_h = jnp.zeros_like(y[0])
        init_c = jnp.zeros_like(y[0])
        init_carry = (init_h, init_c)

        # Use scan to process sequence
        _, losses = jax.lax.scan(single_step, init_carry, (x, y))

        # Sum up losses across all time steps
        return jnp.sum(losses)
    
    def backward(self,xs,ys,epoch=10,lr=0.01):
        Lossgrads = jax.value_and_grad(LSTM.totalLoss,argnums=0)
        optimizer = optax.adam(learning_rate=lr)
        opt_state = optimizer.init(self.params())
        
        for _ in range(epoch):
            value,g = Lossgrads(params,xs,ys)
            print(value)
            updates, opt_state = optimizer.update(g, opt_state)
            params = optax.apply_updates(params, updates)
        
                
    