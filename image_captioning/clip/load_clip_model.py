def load_WavCaps(cuda_available,
                 device,
                 pt_file):
    from ruamel import yaml
    from WavCaps.retrieval.models.ase_model import ASE
    from WavCaps.retrieval.data_handling.text_transform import text_preprocess
    from torch import load

    with open("/home/sfauth/code/MAGIC/image_captioning/clip/WavCaps/retrieval/settings/inference.yaml", "r") as f:
        config = yaml.safe_load(f)

    clip = ASE(config)
    if cuda_available:
        clip = clip.to(device)  
    cp_path = pt_file
    cp = load(cp_path)
    clip.load_state_dict(cp['model'])
    clip.eval()

    return clip


def load_AudioClip(cuda_available,
                 device,
                 pt_file):
    
    from AudioCLIP.model import AudioCLIP

    clip = AudioCLIP(pretrained=pt_file)
    if cuda_available:
        clip = clip.to(device)  
    clip.eval()
    # DISCLAIMER: in the demo, eval mode is not activated!

    return clip


def load_CLAP(cuda_available,
                 device,
                 pt_file):
    
    import laion_clap

    if cuda_available:
        clip = laion_clap.CLAP_Module(enable_fusion=True, device=device)

    else:
        clip = laion_clap.CLAP_Module(enable_fusion=True, device='cpu')

    # create encode_text and encode_audio function

    clip.load_ckpt(ckpt=pt_file,
                   model_id=3)
    clip.eval()
    # DISCLAIMER: in the package example, eval mode is not activated!

    setattr(clip, "encode_audio", get_audio_embedding_from_data)

    return clip