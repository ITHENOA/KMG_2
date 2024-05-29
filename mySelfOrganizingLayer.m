classdef mySelfOrganizingLayer < nnet.layer.Layer & nnet.layer.Formattable
        % & nnet.layer.Acceleratable

    properties (Access=public)
        % (Optional) Layer properties.

        % Declare layer properties here.
        firstBatch = true
    end

    properties (Learnable)
        % (Optional) Layer learnable parameters.

        % Declare learnable parameters here.
        Weights
        Bias
    end

    % properties (State)
    %     % (Optional) Layer state parameters.
    % 
    %     % Declare state parameters here.
    % end

    % properties (Learnable, State)
    %     % (Optional) Nested dlnetwork objects with both learnable
    %     % parameters and state parameters.
    % 
    %     % Declare nested networks with learnable and state parameters here.
    % end

    methods
        function layer = mySelfOrganizingLayer(numInputs, numOutputs, name)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.

            % Define layer constructor function here.
            layer.Name = name;

            % Set layer description.
            layer.Description = "Self-Organizing Layer";

            % Initialize learnable parameters.
            layer.Weights = randn([numOutputs, numInputs], 'single') * 0.01;
            layer.Bias = zeros([numOutputs, 1], 'single');
        end

        % function layer = initialize(layer,layout)
        %     % (Optional) Initialize layer learnable and state parameters.
        %     %
        %     % Inputs:
        %     %         layer  - Layer to initialize
        %     %         layout - Data layout, specified as a networkDataLayout
        %     %                  object
        %     %
        %     % Outputs:
        %     %         layer - Initialized layer
        %     %
        %     %  - For layers with multiple inputs, replace layout with 
        %     %    layout1,...,layoutN, where N is the number of inputs.
        % 
        %     % Define layer initialization function here.
        % end
        

        function [Z] = predict(layer,X)
            % Forward input data through the layer at prediction time and
            % output the result and updated state.
            %
            % Inputs:
            %         layer - Layer to forward propagate through 
            %         X     - Input data
            % Outputs:
            %         Z     - Output of layer forward function
            %         state - (Optional) Updated layer state
            %
            %  - For layers with multiple inputs, replace X with X1,...,XN, 
            %    where N is the number of inputs.
            %  - For layers with multiple outputs, replace Z with 
            %    Z1,...,ZM, where M is the number of outputs.
            %  - For layers with multiple state parameters, replace state 
            %    with state1,...,stateK, where K is the number of state 
            %    parameters.

            % Define layer predict function here.
            Z = fullyconnect(X,layer.Weights,layer.Bias);
        end

        function [Z] = forward(layer,X)
            % (Optional) Forward input data through the layer at training
            % time and output the result, the updated state, and a memory
            % value.
            %
            % Inputs:
            %         layer - Layer to forward propagate through 
            %         X     - Layer input data
            % Outputs:
            %         Z      - Output of layer forward function 
            %         state  - (Optional) Updated layer state 
            %         memory - (Optional) Memory value for custom backward
            %                  function
            %
            %  - For layers with multiple inputs, replace X with X1,...,XN, 
            %    where N is the number of inputs.
            %  - For layers with multiple outputs, replace Z with 
            %    Z1,...,ZM, where M is the number of outputs.
            %  - For layers with multiple state parameters, replace state 
            %    with state1,...,stateK, where K is the number of state 
            %    parameters.

            %%% Define layer forward function here.
            if layer.firstBatch
                % Stage(0): Initialization
                % initPars()
            else
                % Stage(1): Update consequent parameters
                % updatePars()
            end

            Z = fullyconnect(X,layer.Weights,layer.Bias);
            % Z = Z * lambdaMTX;
        end
        
        % function initPars()
        % end
        
        % function updatePars()
        % end

        % function layer = resetState(layer)
        %     % (Optional) Reset layer state.
        % 
        %     % Define reset state function here.
        % end

        % function [dLdX,dLdW,dLdSin] = backward(layer,X,Z,dLdZ,dLdSout,memory)
        %     % (Optional) Backward propagate the derivative of the loss
        %     % function through the layer.
        %     %
        %     % Inputs:
        %     %         layer   - Layer to backward propagate through 
        %     %         X       - Layer input data 
        %     %         Z       - Layer output data 
        %     %         dLdZ    - Derivative of loss with respect to layer 
        %     %                   output
        %     %         dLdSout - (Optional) Derivative of loss with respect 
        %     %                   to state output
        %     %         memory  - Memory value from forward function
        %     % Outputs:
        %     %         dLdX   - Derivative of loss with respect to layer input
        %     %         dLdW   - (Optional) Derivative of loss with respect to
        %     %                  learnable parameter 
        %     %         dLdSin - (Optional) Derivative of loss with respect to 
        %     %                  state input
        %     %
        %     %  - For layers with state parameters, the backward syntax must
        %     %    include both dLdSout and dLdSin, or neither.
        %     %  - For layers with multiple inputs, replace X and dLdX with
        %     %    X1,...,XN and dLdX1,...,dLdXN, respectively, where N is
        %     %    the number of inputs.
        %     %  - For layers with multiple outputs, replace Z and dlZ with
        %     %    Z1,...,ZM and dLdZ,...,dLdZM, respectively, where M is the
        %     %    number of outputs.
        %     %  - For layers with multiple learnable parameters, replace 
        %     %    dLdW with dLdW1,...,dLdWP, where P is the number of 
        %     %    learnable parameters.
        %     %  - For layers with multiple state parameters, replace dLdSin
        %     %    and dLdSout with dLdSin1,...,dLdSinK and 
        %     %    dLdSout1,...,dldSoutK, respectively, where K is the number
        %     %    of state parameters.
        % 
        %     % Define layer backward function here.
        % end
    end
end