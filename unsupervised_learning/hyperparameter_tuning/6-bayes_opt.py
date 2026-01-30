#!/usr/bin/env python3
"""
Optimizes a machine learning model using GPyOpt.
"""
import GPyOpt
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# Set seeds for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

def create_model(learning_rate, units, dropout_rate, l2_reg):
    """
    Creates a simple MLP model for MNIST.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(units, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def run_optimization():
    """
    Runs the Bayesian Optimization.
    """
    # Load Data (MNIST)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Define the bounds of the hyperparameters
    # 1. Learning Rate (Continuous)
    # 2. Units (Discrete)
    # 3. Dropout Rate (Continuous)
    # 4. L2 Regularization (Continuous)
    # 5. Batch Size (Discrete)
    bounds = [
        {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0001, 0.01)},
        {'name': 'units', 'type': 'discrete', 'domain': (32, 64, 128, 256)},
        {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.1, 0.5)},
        {'name': 'l2_reg', 'type': 'continuous', 'domain': (0.0001, 0.01)},
        {'name': 'batch_size', 'type': 'discrete', 'domain': (32, 64, 128)}
    ]

    def objective_function(x):
        """
        The objective function to minimize.
        GPyOpt passes a 2D numpy array.
        """
        # Extract hyperparameters
        # x is shape (1, 5)
        param_values = x[0]
        lr = param_values[0]
        units = int(param_values[1])
        dropout = param_values[2]
        l2 = param_values[3]
        batch_size = int(param_values[4])

        print(f"Training with: lr={lr:.5f}, units={units}, dropout={dropout:.3f}, l2={l2:.5f}, batch={batch_size}")

        model = create_model(lr, units, dropout, l2)

        # Checkpoint filename with hyperparameters
        checkpoint_path = (
            f"best_model_lr-{lr:.5f}_units-{units}_" 
            f"drop-{dropout:.3f}_l2-{l2:.5f}_batch-{batch_size}.h5"
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                               monitor='val_accuracy',
                                               save_best_only=True,
                                               mode='max')
        ]

        history = model.fit(x_train, y_train,
                            epochs=20, # Max epochs, EarlyStopping will cut it short
                            batch_size=batch_size,
                            validation_split=0.2,
                            callbacks=callbacks,
                            verbose=0) # Reduce verbosity for cleaner output

        # Metric to optimize: Validation Accuracy
        # We want to maximize accuracy, but GPyOpt minimizes.
        # So we return 1 - max_val_accuracy (or -max_val_accuracy)
        best_acc = max(history.history['val_accuracy'])
        print(f"Best Validation Accuracy: {best_acc}")
        
        return 1.0 - best_acc

    # Run Bayesian Optimization
    # maximize=False because we are minimizing (1 - accuracy)
    optimizer = GPyOpt.methods.BayesianOptimization(
        f=objective_function,
        domain=bounds,
        model_type='GP',
        acquisition_type='EI', # Expected Improvement
        maximize=False
    )

    optimizer.run_optimization(max_iter=30)

    # Plot Convergence
    optimizer.plot_convergence(filename='convergence_plot.png')
    # Since plot_convergence might just show it, we ensure it saves if the library supports filename
    # GPyOpt's plot_convergence usually displays. If it doesn't save, we can save the data manually.
    # But usually creating a figure before and saving works, or standard GPyOpt usage.
    # Note: GPyOpt internal plotting might not save automatically without tweaking, 
    # but let's assume standard behavior or manually plot if needed. 
    # To be safe, let's manually plot the convergence from optimizer.Y
    
    plt.figure()
    plt.plot(optimizer.Y_best, 'b-')
    plt.xlabel('Iteration')
    plt.ylabel('Best Minimum (1 - Accuracy)')
    plt.title('Convergence of Bayesian Optimization')
    plt.savefig('convergence_plot.png')
    plt.close()


    # Save Report
    with open('bayes_opt.txt', 'w') as f:
        f.write("Bayesian Optimization Report\n")
        f.write("============================\n")
        f.write(f"Best Function Value (1 - Accuracy): {optimizer.fx_opt:.5f}\n")
        f.write(f"Best Accuracy: {1.0 - optimizer.fx_opt:.5f}\n")
        f.write("Best Hyperparameters:\n")
        # Map indices back to names
        # x_opt is a 1D array of the best parameters
        best_params = optimizer.x_opt
        f.write(f"Learning Rate: {best_params[0]:.6f}\n")
        f.write(f"Units: {int(best_params[1])}\n")
        f.write(f"Dropout Rate: {best_params[2]:.6f}\n")
        f.write(f"L2 Regularization: {best_params[3]:.6f}\n")
        f.write(f"Batch Size: {int(best_params[4])}\n")

if __name__ == '__main__':
    run_optimization()
