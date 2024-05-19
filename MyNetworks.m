function Layers = MyNetworks(imgSize,nClass,type)
% arguments
%     imgSize
%     nClass
%     type.
% end
switch type
    % ---------------------------------------------------------------------
    case 1
        Layers = [
            imageInputLayer(imgSize)

            convolution2dLayer(3,32,Padding="same")
            reluLayer
            maxPooling2dLayer(2,"Stride",2)

            convolution2dLayer(3,64,Padding="same")
            reluLayer
            maxPooling2dLayer(2,"Stride",2)

            fullyConnectedLayer(nClass)
            softmaxLayer
            classificationLayer];
        
    % ---------------------------------------------------------------------    
    case "MobileNetv2"
        params = load("MobileNet_params.mat");
        net = dlnetwork;
        tempNet = [
            imageInputLayer([224 224 3],"Name","input_1","Normalization","zscore","Mean",params.input_1.Mean,"StandardDeviation",params.input_1.StandardDeviation)
            convolution2dLayer([3 3],32,"Name","Conv1","Padding","same","Stride",[2 2],"Bias",params.Conv1.Bias,"Weights",params.Conv1.Weights)
            batchNormalizationLayer("Name","bn_Conv1","Epsilon",0.001,"Offset",params.bn_Conv1.Offset,"Scale",params.bn_Conv1.Scale,"TrainedMean",params.bn_Conv1.TrainedMean,"TrainedVariance",params.bn_Conv1.TrainedVariance)
            clippedReluLayer(6,"Name","Conv1_relu")
            groupedConvolution2dLayer([3 3],1,32,"Name","expanded_conv_depthwise","Padding","same","Bias",params.expanded_conv_depthwise.Bias,"Weights",params.expanded_conv_depthwise.Weights)
            batchNormalizationLayer("Name","expanded_conv_depthwise_BN","Epsilon",0.001,"Offset",params.expanded_conv_depthwise_BN.Offset,"Scale",params.expanded_conv_depthwise_BN.Scale,"TrainedMean",params.expanded_conv_depthwise_BN.TrainedMean,"TrainedVariance",params.expanded_conv_depthwise_BN.TrainedVariance)
            clippedReluLayer(6,"Name","expanded_conv_depthwise_relu")
            convolution2dLayer([1 1],16,"Name","expanded_conv_project","Padding","same","Bias",params.expanded_conv_project.Bias,"Weights",params.expanded_conv_project.Weights)
            batchNormalizationLayer("Name","expanded_conv_project_BN","Epsilon",0.001,"Offset",params.expanded_conv_project_BN.Offset,"Scale",params.expanded_conv_project_BN.Scale,"TrainedMean",params.expanded_conv_project_BN.TrainedMean,"TrainedVariance",params.expanded_conv_project_BN.TrainedVariance)
            convolution2dLayer([1 1],96,"Name","block_1_expand","Padding","same","Bias",params.block_1_expand.Bias,"Weights",params.block_1_expand.Weights)
            batchNormalizationLayer("Name","block_1_expand_BN","Epsilon",0.001,"Offset",params.block_1_expand_BN.Offset,"Scale",params.block_1_expand_BN.Scale,"TrainedMean",params.block_1_expand_BN.TrainedMean,"TrainedVariance",params.block_1_expand_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_1_expand_relu")
            groupedConvolution2dLayer([3 3],1,96,"Name","block_1_depthwise","Padding","same","Stride",[2 2],"Bias",params.block_1_depthwise.Bias,"Weights",params.block_1_depthwise.Weights)
            batchNormalizationLayer("Name","block_1_depthwise_BN","Epsilon",0.001,"Offset",params.block_1_depthwise_BN.Offset,"Scale",params.block_1_depthwise_BN.Scale,"TrainedMean",params.block_1_depthwise_BN.TrainedMean,"TrainedVariance",params.block_1_depthwise_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_1_depthwise_relu")
            convolution2dLayer([1 1],24,"Name","block_1_project","Padding","same","Bias",params.block_1_project.Bias,"Weights",params.block_1_project.Weights)
            batchNormalizationLayer("Name","block_1_project_BN","Epsilon",0.001,"Offset",params.block_1_project_BN.Offset,"Scale",params.block_1_project_BN.Scale,"TrainedMean",params.block_1_project_BN.TrainedMean,"TrainedVariance",params.block_1_project_BN.TrainedVariance)];
        net = addLayers(net,tempNet);

        tempNet = [
            convolution2dLayer([1 1],144,"Name","block_2_expand","Padding","same","Bias",params.block_2_expand.Bias,"Weights",params.block_2_expand.Weights)
            batchNormalizationLayer("Name","block_2_expand_BN","Epsilon",0.001,"Offset",params.block_2_expand_BN.Offset,"Scale",params.block_2_expand_BN.Scale,"TrainedMean",params.block_2_expand_BN.TrainedMean,"TrainedVariance",params.block_2_expand_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_2_expand_relu")
            groupedConvolution2dLayer([3 3],1,144,"Name","block_2_depthwise","Padding","same","Bias",params.block_2_depthwise.Bias,"Weights",params.block_2_depthwise.Weights)
            batchNormalizationLayer("Name","block_2_depthwise_BN","Epsilon",0.001,"Offset",params.block_2_depthwise_BN.Offset,"Scale",params.block_2_depthwise_BN.Scale,"TrainedMean",params.block_2_depthwise_BN.TrainedMean,"TrainedVariance",params.block_2_depthwise_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_2_depthwise_relu")
            convolution2dLayer([1 1],24,"Name","block_2_project","Padding","same","Bias",params.block_2_project.Bias,"Weights",params.block_2_project.Weights)
            batchNormalizationLayer("Name","block_2_project_BN","Epsilon",0.001,"Offset",params.block_2_project_BN.Offset,"Scale",params.block_2_project_BN.Scale,"TrainedMean",params.block_2_project_BN.TrainedMean,"TrainedVariance",params.block_2_project_BN.TrainedVariance)];
        net = addLayers(net,tempNet);

        tempNet = [
            additionLayer(2,"Name","block_2_add")
            convolution2dLayer([1 1],144,"Name","block_3_expand","Padding","same","Bias",params.block_3_expand.Bias,"Weights",params.block_3_expand.Weights)
            batchNormalizationLayer("Name","block_3_expand_BN","Epsilon",0.001,"Offset",params.block_3_expand_BN.Offset,"Scale",params.block_3_expand_BN.Scale,"TrainedMean",params.block_3_expand_BN.TrainedMean,"TrainedVariance",params.block_3_expand_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_3_expand_relu")
            groupedConvolution2dLayer([3 3],1,144,"Name","block_3_depthwise","Padding","same","Stride",[2 2],"Bias",params.block_3_depthwise.Bias,"Weights",params.block_3_depthwise.Weights)
            batchNormalizationLayer("Name","block_3_depthwise_BN","Epsilon",0.001,"Offset",params.block_3_depthwise_BN.Offset,"Scale",params.block_3_depthwise_BN.Scale,"TrainedMean",params.block_3_depthwise_BN.TrainedMean,"TrainedVariance",params.block_3_depthwise_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_3_depthwise_relu")
            convolution2dLayer([1 1],32,"Name","block_3_project","Padding","same","Bias",params.block_3_project.Bias,"Weights",params.block_3_project.Weights)
            batchNormalizationLayer("Name","block_3_project_BN","Epsilon",0.001,"Offset",params.block_3_project_BN.Offset,"Scale",params.block_3_project_BN.Scale,"TrainedMean",params.block_3_project_BN.TrainedMean,"TrainedVariance",params.block_3_project_BN.TrainedVariance)];
        net = addLayers(net,tempNet);

        tempNet = [
            convolution2dLayer([1 1],192,"Name","block_4_expand","Padding","same","Bias",params.block_4_expand.Bias,"Weights",params.block_4_expand.Weights)
            batchNormalizationLayer("Name","block_4_expand_BN","Epsilon",0.001,"Offset",params.block_4_expand_BN.Offset,"Scale",params.block_4_expand_BN.Scale,"TrainedMean",params.block_4_expand_BN.TrainedMean,"TrainedVariance",params.block_4_expand_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_4_expand_relu")
            groupedConvolution2dLayer([3 3],1,192,"Name","block_4_depthwise","Padding","same","Bias",params.block_4_depthwise.Bias,"Weights",params.block_4_depthwise.Weights)
            batchNormalizationLayer("Name","block_4_depthwise_BN","Epsilon",0.001,"Offset",params.block_4_depthwise_BN.Offset,"Scale",params.block_4_depthwise_BN.Scale,"TrainedMean",params.block_4_depthwise_BN.TrainedMean,"TrainedVariance",params.block_4_depthwise_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_4_depthwise_relu")
            convolution2dLayer([1 1],32,"Name","block_4_project","Padding","same","Bias",params.block_4_project.Bias,"Weights",params.block_4_project.Weights)
            batchNormalizationLayer("Name","block_4_project_BN","Epsilon",0.001,"Offset",params.block_4_project_BN.Offset,"Scale",params.block_4_project_BN.Scale,"TrainedMean",params.block_4_project_BN.TrainedMean,"TrainedVariance",params.block_4_project_BN.TrainedVariance)];
        net = addLayers(net,tempNet);

        tempNet = additionLayer(2,"Name","block_4_add");
        net = addLayers(net,tempNet);

        tempNet = [
            convolution2dLayer([1 1],192,"Name","block_5_expand","Padding","same","Bias",params.block_5_expand.Bias,"Weights",params.block_5_expand.Weights)
            batchNormalizationLayer("Name","block_5_expand_BN","Epsilon",0.001,"Offset",params.block_5_expand_BN.Offset,"Scale",params.block_5_expand_BN.Scale,"TrainedMean",params.block_5_expand_BN.TrainedMean,"TrainedVariance",params.block_5_expand_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_5_expand_relu")
            groupedConvolution2dLayer([3 3],1,192,"Name","block_5_depthwise","Padding","same","Bias",params.block_5_depthwise.Bias,"Weights",params.block_5_depthwise.Weights)
            batchNormalizationLayer("Name","block_5_depthwise_BN","Epsilon",0.001,"Offset",params.block_5_depthwise_BN.Offset,"Scale",params.block_5_depthwise_BN.Scale,"TrainedMean",params.block_5_depthwise_BN.TrainedMean,"TrainedVariance",params.block_5_depthwise_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_5_depthwise_relu")
            convolution2dLayer([1 1],32,"Name","block_5_project","Padding","same","Bias",params.block_5_project.Bias,"Weights",params.block_5_project.Weights)
            batchNormalizationLayer("Name","block_5_project_BN","Epsilon",0.001,"Offset",params.block_5_project_BN.Offset,"Scale",params.block_5_project_BN.Scale,"TrainedMean",params.block_5_project_BN.TrainedMean,"TrainedVariance",params.block_5_project_BN.TrainedVariance)];
        net = addLayers(net,tempNet);

        tempNet = [
            additionLayer(2,"Name","block_5_add")
            convolution2dLayer([1 1],192,"Name","block_6_expand","Padding","same","Bias",params.block_6_expand.Bias,"Weights",params.block_6_expand.Weights)
            batchNormalizationLayer("Name","block_6_expand_BN","Epsilon",0.001,"Offset",params.block_6_expand_BN.Offset,"Scale",params.block_6_expand_BN.Scale,"TrainedMean",params.block_6_expand_BN.TrainedMean,"TrainedVariance",params.block_6_expand_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_6_expand_relu")
            groupedConvolution2dLayer([3 3],1,192,"Name","block_6_depthwise","Padding","same","Stride",[2 2],"Bias",params.block_6_depthwise.Bias,"Weights",params.block_6_depthwise.Weights)
            batchNormalizationLayer("Name","block_6_depthwise_BN","Epsilon",0.001,"Offset",params.block_6_depthwise_BN.Offset,"Scale",params.block_6_depthwise_BN.Scale,"TrainedMean",params.block_6_depthwise_BN.TrainedMean,"TrainedVariance",params.block_6_depthwise_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_6_depthwise_relu")
            convolution2dLayer([1 1],64,"Name","block_6_project","Padding","same","Bias",params.block_6_project.Bias,"Weights",params.block_6_project.Weights)
            batchNormalizationLayer("Name","block_6_project_BN","Epsilon",0.001,"Offset",params.block_6_project_BN.Offset,"Scale",params.block_6_project_BN.Scale,"TrainedMean",params.block_6_project_BN.TrainedMean,"TrainedVariance",params.block_6_project_BN.TrainedVariance)];
        net = addLayers(net,tempNet);

        tempNet = [
            convolution2dLayer([1 1],384,"Name","block_7_expand","Padding","same","Bias",params.block_7_expand.Bias,"Weights",params.block_7_expand.Weights)
            batchNormalizationLayer("Name","block_7_expand_BN","Epsilon",0.001,"Offset",params.block_7_expand_BN.Offset,"Scale",params.block_7_expand_BN.Scale,"TrainedMean",params.block_7_expand_BN.TrainedMean,"TrainedVariance",params.block_7_expand_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_7_expand_relu")
            groupedConvolution2dLayer([3 3],1,384,"Name","block_7_depthwise","Padding","same","Bias",params.block_7_depthwise.Bias,"Weights",params.block_7_depthwise.Weights)
            batchNormalizationLayer("Name","block_7_depthwise_BN","Epsilon",0.001,"Offset",params.block_7_depthwise_BN.Offset,"Scale",params.block_7_depthwise_BN.Scale,"TrainedMean",params.block_7_depthwise_BN.TrainedMean,"TrainedVariance",params.block_7_depthwise_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_7_depthwise_relu")
            convolution2dLayer([1 1],64,"Name","block_7_project","Padding","same","Bias",params.block_7_project.Bias,"Weights",params.block_7_project.Weights)
            batchNormalizationLayer("Name","block_7_project_BN","Epsilon",0.001,"Offset",params.block_7_project_BN.Offset,"Scale",params.block_7_project_BN.Scale,"TrainedMean",params.block_7_project_BN.TrainedMean,"TrainedVariance",params.block_7_project_BN.TrainedVariance)];
        net = addLayers(net,tempNet);

        tempNet = additionLayer(2,"Name","block_7_add");
        net = addLayers(net,tempNet);

        tempNet = [
            convolution2dLayer([1 1],384,"Name","block_8_expand","Padding","same","Bias",params.block_8_expand.Bias,"Weights",params.block_8_expand.Weights)
            batchNormalizationLayer("Name","block_8_expand_BN","Epsilon",0.001,"Offset",params.block_8_expand_BN.Offset,"Scale",params.block_8_expand_BN.Scale,"TrainedMean",params.block_8_expand_BN.TrainedMean,"TrainedVariance",params.block_8_expand_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_8_expand_relu")
            groupedConvolution2dLayer([3 3],1,384,"Name","block_8_depthwise","Padding","same","Bias",params.block_8_depthwise.Bias,"Weights",params.block_8_depthwise.Weights)
            batchNormalizationLayer("Name","block_8_depthwise_BN","Epsilon",0.001,"Offset",params.block_8_depthwise_BN.Offset,"Scale",params.block_8_depthwise_BN.Scale,"TrainedMean",params.block_8_depthwise_BN.TrainedMean,"TrainedVariance",params.block_8_depthwise_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_8_depthwise_relu")
            convolution2dLayer([1 1],64,"Name","block_8_project","Padding","same","Bias",params.block_8_project.Bias,"Weights",params.block_8_project.Weights)
            batchNormalizationLayer("Name","block_8_project_BN","Epsilon",0.001,"Offset",params.block_8_project_BN.Offset,"Scale",params.block_8_project_BN.Scale,"TrainedMean",params.block_8_project_BN.TrainedMean,"TrainedVariance",params.block_8_project_BN.TrainedVariance)];
        net = addLayers(net,tempNet);

        tempNet = additionLayer(2,"Name","block_8_add");
        net = addLayers(net,tempNet);

        tempNet = [
            convolution2dLayer([1 1],384,"Name","block_9_expand","Padding","same","Bias",params.block_9_expand.Bias,"Weights",params.block_9_expand.Weights)
            batchNormalizationLayer("Name","block_9_expand_BN","Epsilon",0.001,"Offset",params.block_9_expand_BN.Offset,"Scale",params.block_9_expand_BN.Scale,"TrainedMean",params.block_9_expand_BN.TrainedMean,"TrainedVariance",params.block_9_expand_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_9_expand_relu")
            groupedConvolution2dLayer([3 3],1,384,"Name","block_9_depthwise","Padding","same","Bias",params.block_9_depthwise.Bias,"Weights",params.block_9_depthwise.Weights)
            batchNormalizationLayer("Name","block_9_depthwise_BN","Epsilon",0.001,"Offset",params.block_9_depthwise_BN.Offset,"Scale",params.block_9_depthwise_BN.Scale,"TrainedMean",params.block_9_depthwise_BN.TrainedMean,"TrainedVariance",params.block_9_depthwise_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_9_depthwise_relu")
            convolution2dLayer([1 1],64,"Name","block_9_project","Padding","same","Bias",params.block_9_project.Bias,"Weights",params.block_9_project.Weights)
            batchNormalizationLayer("Name","block_9_project_BN","Epsilon",0.001,"Offset",params.block_9_project_BN.Offset,"Scale",params.block_9_project_BN.Scale,"TrainedMean",params.block_9_project_BN.TrainedMean,"TrainedVariance",params.block_9_project_BN.TrainedVariance)];
        net = addLayers(net,tempNet);

        tempNet = [
            additionLayer(2,"Name","block_9_add")
            convolution2dLayer([1 1],384,"Name","block_10_expand","Padding","same","Bias",params.block_10_expand.Bias,"Weights",params.block_10_expand.Weights)
            batchNormalizationLayer("Name","block_10_expand_BN","Epsilon",0.001,"Offset",params.block_10_expand_BN.Offset,"Scale",params.block_10_expand_BN.Scale,"TrainedMean",params.block_10_expand_BN.TrainedMean,"TrainedVariance",params.block_10_expand_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_10_expand_relu")
            groupedConvolution2dLayer([3 3],1,384,"Name","block_10_depthwise","Padding","same","Bias",params.block_10_depthwise.Bias,"Weights",params.block_10_depthwise.Weights)
            batchNormalizationLayer("Name","block_10_depthwise_BN","Epsilon",0.001,"Offset",params.block_10_depthwise_BN.Offset,"Scale",params.block_10_depthwise_BN.Scale,"TrainedMean",params.block_10_depthwise_BN.TrainedMean,"TrainedVariance",params.block_10_depthwise_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_10_depthwise_relu")
            convolution2dLayer([1 1],96,"Name","block_10_project","Padding","same","Bias",params.block_10_project.Bias,"Weights",params.block_10_project.Weights)
            batchNormalizationLayer("Name","block_10_project_BN","Epsilon",0.001,"Offset",params.block_10_project_BN.Offset,"Scale",params.block_10_project_BN.Scale,"TrainedMean",params.block_10_project_BN.TrainedMean,"TrainedVariance",params.block_10_project_BN.TrainedVariance)];
        net = addLayers(net,tempNet);

        tempNet = [
            convolution2dLayer([1 1],576,"Name","block_11_expand","Padding","same","Bias",params.block_11_expand.Bias,"Weights",params.block_11_expand.Weights)
            batchNormalizationLayer("Name","block_11_expand_BN","Epsilon",0.001,"Offset",params.block_11_expand_BN.Offset,"Scale",params.block_11_expand_BN.Scale,"TrainedMean",params.block_11_expand_BN.TrainedMean,"TrainedVariance",params.block_11_expand_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_11_expand_relu")
            groupedConvolution2dLayer([3 3],1,576,"Name","block_11_depthwise","Padding","same","Bias",params.block_11_depthwise.Bias,"Weights",params.block_11_depthwise.Weights)
            batchNormalizationLayer("Name","block_11_depthwise_BN","Epsilon",0.001,"Offset",params.block_11_depthwise_BN.Offset,"Scale",params.block_11_depthwise_BN.Scale,"TrainedMean",params.block_11_depthwise_BN.TrainedMean,"TrainedVariance",params.block_11_depthwise_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_11_depthwise_relu")
            convolution2dLayer([1 1],96,"Name","block_11_project","Padding","same","Bias",params.block_11_project.Bias,"Weights",params.block_11_project.Weights)
            batchNormalizationLayer("Name","block_11_project_BN","Epsilon",0.001,"Offset",params.block_11_project_BN.Offset,"Scale",params.block_11_project_BN.Scale,"TrainedMean",params.block_11_project_BN.TrainedMean,"TrainedVariance",params.block_11_project_BN.TrainedVariance)];
        net = addLayers(net,tempNet);

        tempNet = additionLayer(2,"Name","block_11_add");
        net = addLayers(net,tempNet);

        tempNet = [
            convolution2dLayer([1 1],576,"Name","block_12_expand","Padding","same","Bias",params.block_12_expand.Bias,"Weights",params.block_12_expand.Weights)
            batchNormalizationLayer("Name","block_12_expand_BN","Epsilon",0.001,"Offset",params.block_12_expand_BN.Offset,"Scale",params.block_12_expand_BN.Scale,"TrainedMean",params.block_12_expand_BN.TrainedMean,"TrainedVariance",params.block_12_expand_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_12_expand_relu")
            groupedConvolution2dLayer([3 3],1,576,"Name","block_12_depthwise","Padding","same","Bias",params.block_12_depthwise.Bias,"Weights",params.block_12_depthwise.Weights)
            batchNormalizationLayer("Name","block_12_depthwise_BN","Epsilon",0.001,"Offset",params.block_12_depthwise_BN.Offset,"Scale",params.block_12_depthwise_BN.Scale,"TrainedMean",params.block_12_depthwise_BN.TrainedMean,"TrainedVariance",params.block_12_depthwise_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_12_depthwise_relu")
            convolution2dLayer([1 1],96,"Name","block_12_project","Padding","same","Bias",params.block_12_project.Bias,"Weights",params.block_12_project.Weights)
            batchNormalizationLayer("Name","block_12_project_BN","Epsilon",0.001,"Offset",params.block_12_project_BN.Offset,"Scale",params.block_12_project_BN.Scale,"TrainedMean",params.block_12_project_BN.TrainedMean,"TrainedVariance",params.block_12_project_BN.TrainedVariance)];
        net = addLayers(net,tempNet);

        tempNet = [
            additionLayer(2,"Name","block_12_add")
            convolution2dLayer([1 1],576,"Name","block_13_expand","Padding","same","Bias",params.block_13_expand.Bias,"Weights",params.block_13_expand.Weights)
            batchNormalizationLayer("Name","block_13_expand_BN","Epsilon",0.001,"Offset",params.block_13_expand_BN.Offset,"Scale",params.block_13_expand_BN.Scale,"TrainedMean",params.block_13_expand_BN.TrainedMean,"TrainedVariance",params.block_13_expand_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_13_expand_relu")
            groupedConvolution2dLayer([3 3],1,576,"Name","block_13_depthwise","Padding","same","Stride",[2 2],"Bias",params.block_13_depthwise.Bias,"Weights",params.block_13_depthwise.Weights)
            batchNormalizationLayer("Name","block_13_depthwise_BN","Epsilon",0.001,"Offset",params.block_13_depthwise_BN.Offset,"Scale",params.block_13_depthwise_BN.Scale,"TrainedMean",params.block_13_depthwise_BN.TrainedMean,"TrainedVariance",params.block_13_depthwise_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_13_depthwise_relu")
            convolution2dLayer([1 1],160,"Name","block_13_project","Padding","same","Bias",params.block_13_project.Bias,"Weights",params.block_13_project.Weights)
            batchNormalizationLayer("Name","block_13_project_BN","Epsilon",0.001,"Offset",params.block_13_project_BN.Offset,"Scale",params.block_13_project_BN.Scale,"TrainedMean",params.block_13_project_BN.TrainedMean,"TrainedVariance",params.block_13_project_BN.TrainedVariance)];
        net = addLayers(net,tempNet);

        tempNet = [
            convolution2dLayer([1 1],960,"Name","block_14_expand","Padding","same","Bias",params.block_14_expand.Bias,"Weights",params.block_14_expand.Weights)
            batchNormalizationLayer("Name","block_14_expand_BN","Epsilon",0.001,"Offset",params.block_14_expand_BN.Offset,"Scale",params.block_14_expand_BN.Scale,"TrainedMean",params.block_14_expand_BN.TrainedMean,"TrainedVariance",params.block_14_expand_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_14_expand_relu")
            groupedConvolution2dLayer([3 3],1,960,"Name","block_14_depthwise","Padding","same","Bias",params.block_14_depthwise.Bias,"Weights",params.block_14_depthwise.Weights)
            batchNormalizationLayer("Name","block_14_depthwise_BN","Epsilon",0.001,"Offset",params.block_14_depthwise_BN.Offset,"Scale",params.block_14_depthwise_BN.Scale,"TrainedMean",params.block_14_depthwise_BN.TrainedMean,"TrainedVariance",params.block_14_depthwise_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_14_depthwise_relu")
            convolution2dLayer([1 1],160,"Name","block_14_project","Padding","same","Bias",params.block_14_project.Bias,"Weights",params.block_14_project.Weights)
            batchNormalizationLayer("Name","block_14_project_BN","Epsilon",0.001,"Offset",params.block_14_project_BN.Offset,"Scale",params.block_14_project_BN.Scale,"TrainedMean",params.block_14_project_BN.TrainedMean,"TrainedVariance",params.block_14_project_BN.TrainedVariance)];
        net = addLayers(net,tempNet);

        tempNet = additionLayer(2,"Name","block_14_add");
        net = addLayers(net,tempNet);

        tempNet = [
            convolution2dLayer([1 1],960,"Name","block_15_expand","Padding","same","Bias",params.block_15_expand.Bias,"Weights",params.block_15_expand.Weights)
            batchNormalizationLayer("Name","block_15_expand_BN","Epsilon",0.001,"Offset",params.block_15_expand_BN.Offset,"Scale",params.block_15_expand_BN.Scale,"TrainedMean",params.block_15_expand_BN.TrainedMean,"TrainedVariance",params.block_15_expand_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_15_expand_relu")
            groupedConvolution2dLayer([3 3],1,960,"Name","block_15_depthwise","Padding","same","Bias",params.block_15_depthwise.Bias,"Weights",params.block_15_depthwise.Weights)
            batchNormalizationLayer("Name","block_15_depthwise_BN","Epsilon",0.001,"Offset",params.block_15_depthwise_BN.Offset,"Scale",params.block_15_depthwise_BN.Scale,"TrainedMean",params.block_15_depthwise_BN.TrainedMean,"TrainedVariance",params.block_15_depthwise_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_15_depthwise_relu")
            convolution2dLayer([1 1],160,"Name","block_15_project","Padding","same","Bias",params.block_15_project.Bias,"Weights",params.block_15_project.Weights)
            batchNormalizationLayer("Name","block_15_project_BN","Epsilon",0.001,"Offset",params.block_15_project_BN.Offset,"Scale",params.block_15_project_BN.Scale,"TrainedMean",params.block_15_project_BN.TrainedMean,"TrainedVariance",params.block_15_project_BN.TrainedVariance)];
        net = addLayers(net,tempNet);

        tempNet = [
            additionLayer(2,"Name","block_15_add")
            convolution2dLayer([1 1],960,"Name","block_16_expand","Padding","same","Bias",params.block_16_expand.Bias,"Weights",params.block_16_expand.Weights)
            batchNormalizationLayer("Name","block_16_expand_BN","Epsilon",0.001,"Offset",params.block_16_expand_BN.Offset,"Scale",params.block_16_expand_BN.Scale,"TrainedMean",params.block_16_expand_BN.TrainedMean,"TrainedVariance",params.block_16_expand_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_16_expand_relu")
            groupedConvolution2dLayer([3 3],1,960,"Name","block_16_depthwise","Padding","same","Bias",params.block_16_depthwise.Bias,"Weights",params.block_16_depthwise.Weights)
            batchNormalizationLayer("Name","block_16_depthwise_BN","Epsilon",0.001,"Offset",params.block_16_depthwise_BN.Offset,"Scale",params.block_16_depthwise_BN.Scale,"TrainedMean",params.block_16_depthwise_BN.TrainedMean,"TrainedVariance",params.block_16_depthwise_BN.TrainedVariance)
            clippedReluLayer(6,"Name","block_16_depthwise_relu")
            convolution2dLayer([1 1],320,"Name","block_16_project","Padding","same","Bias",params.block_16_project.Bias,"Weights",params.block_16_project.Weights)
            batchNormalizationLayer("Name","block_16_project_BN","Epsilon",0.001,"Offset",params.block_16_project_BN.Offset,"Scale",params.block_16_project_BN.Scale,"TrainedMean",params.block_16_project_BN.TrainedMean,"TrainedVariance",params.block_16_project_BN.TrainedVariance)
            convolution2dLayer([1 1],1280,"Name","Conv_1","Bias",params.Conv_1.Bias,"Weights",params.Conv_1.Weights)
            batchNormalizationLayer("Name","Conv_1_bn","Epsilon",0.001,"Offset",params.Conv_1_bn.Offset,"Scale",params.Conv_1_bn.Scale,"TrainedMean",params.Conv_1_bn.TrainedMean,"TrainedVariance",params.Conv_1_bn.TrainedVariance)
            clippedReluLayer(6,"Name","out_relu")
            globalAveragePooling2dLayer("Name","global_average_pooling2d_1")
            fullyConnectedLayer(nClass,"Name","Logits")
            softmaxLayer("Name","Logits_softmax")];
        net = addLayers(net,tempNet);

        % clean up helper variable
        clear tempNet;

        net = connectLayers(net,"block_1_project_BN","block_2_expand");
        net = connectLayers(net,"block_1_project_BN","block_2_add/in2");
        net = connectLayers(net,"block_2_project_BN","block_2_add/in1");
        net = connectLayers(net,"block_3_project_BN","block_4_expand");
        net = connectLayers(net,"block_3_project_BN","block_4_add/in2");
        net = connectLayers(net,"block_4_project_BN","block_4_add/in1");
        net = connectLayers(net,"block_4_add","block_5_expand");
        net = connectLayers(net,"block_4_add","block_5_add/in2");
        net = connectLayers(net,"block_5_project_BN","block_5_add/in1");
        net = connectLayers(net,"block_6_project_BN","block_7_expand");
        net = connectLayers(net,"block_6_project_BN","block_7_add/in2");
        net = connectLayers(net,"block_7_project_BN","block_7_add/in1");
        net = connectLayers(net,"block_7_add","block_8_expand");
        net = connectLayers(net,"block_7_add","block_8_add/in2");
        net = connectLayers(net,"block_8_project_BN","block_8_add/in1");
        net = connectLayers(net,"block_8_add","block_9_expand");
        net = connectLayers(net,"block_8_add","block_9_add/in2");
        net = connectLayers(net,"block_9_project_BN","block_9_add/in1");
        net = connectLayers(net,"block_10_project_BN","block_11_expand");
        net = connectLayers(net,"block_10_project_BN","block_11_add/in2");
        net = connectLayers(net,"block_11_project_BN","block_11_add/in1");
        net = connectLayers(net,"block_11_add","block_12_expand");
        net = connectLayers(net,"block_11_add","block_12_add/in2");
        net = connectLayers(net,"block_12_project_BN","block_12_add/in1");
        net = connectLayers(net,"block_13_project_BN","block_14_expand");
        net = connectLayers(net,"block_13_project_BN","block_14_add/in2");
        net = connectLayers(net,"block_14_project_BN","block_14_add/in1");
        net = connectLayers(net,"block_14_add","block_15_expand");
        net = connectLayers(net,"block_14_add","block_15_add/in2");
        net = connectLayers(net,"block_15_project_BN","block_15_add/in1");
        Layers = initialize(net);
    % ---------------------------------------------------------------------
    case 3
end