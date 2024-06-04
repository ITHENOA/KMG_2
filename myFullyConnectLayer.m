classdef myFullyConnectLayer < nnet.layer.Layer & nnet.layer.Formattable
    properties (Learnable)
        % Layer learnable parameters
        Weights
        Bias
    end
    
    methods
        function layer = myFullyConnectLayer(inputSize, outputSize, name)
            % Set layer name
            layer.Name = name;
            
            % Set layer description
            layer.Description = "Fully connected layer with " + outputSize + " outputs";
            
            % Initialize weights and bias
            layer.Weights = randn([outputSize, inputSize], 'single') * 0.01;
            layer.Bias = zeros([outputSize, 1], 'single');
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result
            
            Z = fullyconnect(X,layer.Weights,layer.Bias);

        end
        function Z = forward(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result
            
            Z = fullyconnect(X,layer.Weights,layer.Bias);

        end
    end
end
