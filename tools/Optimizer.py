import torch.optim as optim

class Optimizer:
    def __init__(self, params, args):        
        """
        Initializes the optimizer and learning rate scheduler.

        Args:
            params: Parameters of the model to optimize.
            args: Arguments containing hyperparameters such as learning rate, weight decay, and scheduler settings.
        """
        self.optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.w_dc)
        self.set_scheduler(args)
        
    def zero_grad(self):
        """
        Clears the gradients of all optimized tensors.
        """
        self.optimizer.zero_grad()

    def step(self):
        """
        Performs a single optimization step.
        """
        self.optimizer.step()
        
    def set_scheduler(self, args):
        """
        Sets the learning rate scheduler with the provided arguments.

        Args:
            args: Arguments containing the learning rate decay step and gamma value.
        """
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=args.lr_dc_step,
                                                   gamma=args.lr_dc)
        
    def step_scheduler(self):
        """
        Updates the learning rate using the scheduler.
        """
        self.scheduler.step()
