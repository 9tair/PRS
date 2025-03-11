def freeze_final_layer(model, modelname, logger):
    """Freeze the parameters of the final layer in the model."""
    if 'resnet' in modelname.lower():
        final_layer = model.fc
    elif 'vgg' in modelname.lower():
        final_layer = model.classifier[-1]
    elif 'mobilenet' in modelname.lower():
        final_layer = model.classifier
    else:
        try:
            final_layer = model.fc
            logger.info(f"Using model.fc as final layer for {modelname}")
        except AttributeError:
            final_layer = None
            for name, module in reversed(list(model.named_modules())):
                if len(list(module.parameters())) > 0:
                    final_layer = module
                    logger.info(f"Identified final layer as: {name}")
                    break

    if final_layer:
        for param in final_layer.parameters():
            param.requires_grad = False
        logger.info(f"Frozen parameters of the final layer for {modelname}")
    else:
        logger.error(f"Could not find final layer to freeze for {modelname}")
