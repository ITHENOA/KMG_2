classdef customFullyConnectedLayer < nnet.layer.Layer
    properties (Learnable)
        % Learnable weights and bias
        Weights
        Bias
    end
    
    properties
        % Mask to indicate which weights are learnable
        LearnableMask
    end
    
    methods
        function layer = customFullyConnectedLayer(numInputs, numOutputs, learnableMask, name)
            % Create a custom fully connected layer.
            
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = "Custom fully connected layer with " + numInputs + " inputs and " + numOutputs + " outputs";
            
            % Initialize learnable parameters.
            layer.Weights = randn([numOutputs, numInputs], 'single');
            layer.Bias = randn([numOutputs, 1], 'single');
            
            % Set learnable mask.
            layer.LearnableMask = learnableMask;
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.

            % Multiply by weights and add bias.
            Z = fullyconnect(X, layer.Weights, layer.Bias);
        end
        
        function [Z, memory] = forward(layer, X)
            % Forward input data through the layer at training time and
            % output the result.

            % Multiply by weights and add bias.
            Z = fullyconnect(X, layer.Weights, layer.Bias);
            memory = [];
        end
        
        function [dLdX, dLdW, dLdB] = backward(layer, X, Z, dLdZ, memory)
            % Backward propagate the derivative of the loss function through 
            % the layer.
            
            % Compute gradients of the weights and biases.
            dLdX = layer.Weights' * dLdZ;
            dLdW = dLdZ * X';
            dLdB = sum(dLdZ, 2);
            
            % Apply the learnable mask to the weight gradients.
            dLdW = dLdW .* layer.LearnableMask;
        end
    end
end

function Z = fullyconnect(X, weights, bias)
    % Custom fully connected operation.
    Z = weights * X + bias;
end
%%

classdef customFullyConnectedLayer < nnet.layer.Layer
    properties
        % Built-in fully connected layer
        InternalLayer
        
        % Mask to indicate which weights are learnable
        LearnableMask
        
        % Frozen weights
        FrozenWeights
    end
    
    properties (Learnable)
        % Learnable parameters
        Weights
        Bias
    end
    
    methods
        function layer = customFullyConnectedLayer(numInputs, numOutputs, learnableMask, name)
            % Create a custom fully connected layer.
            
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = "Custom fully connected layer with " + numInputs + " inputs and " + numOutputs + " outputs";
            
            % Create the built-in fully connected layer
            layer.InternalLayer = fullyConnectedLayer(numOutputs, 'Name', 'internal_fc');

            % Initialize learnable parameters
            layer.Weights = layer.InternalLayer.Weights;
            layer.Bias = layer.InternalLayer.Bias;
            
            % Set learnable mask and frozen weights
            layer.LearnableMask = dlarray(single(learnableMask));
            layer.FrozenWeights = layer.Weights .* ~layer.LearnableMask;
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.

            % Combine learnable and frozen weights
            combinedWeights = layer.Weights .* layer.LearnableMask + layer.FrozenWeights;
            
            % Set the internal layer weights and bias
            layer.InternalLayer.Weights = combinedWeights;
            layer.InternalLayer.Bias = layer.Bias;
            
            % Use the built-in fully connected layer for prediction
            Z = predict(layer.InternalLayer, X);
        end
        
        function [Z, memory] = forward(layer, X)
            % Forward input data through the layer at training time and
            % output the result.

            % Combine learnable and frozen weights
            combinedWeights = layer.Weights .* layer.LearnableMask + layer.FrozenWeights;
            
            % Set the internal layer weights and bias
            layer.InternalLayer.Weights = combinedWeights;
            layer.InternalLayer.Bias = layer.Bias;
            
            % Use the built-in fully connected layer for forward pass
            [Z, memory] = forward(layer.InternalLayer, X);
        end
    end
end

