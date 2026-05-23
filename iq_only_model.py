# I/Q-Only variant: same architecture but with SNR conditioning disabled
# Set snr_cond=None in all layers, or replace SNR embedding with zeros

def build_iq_only_scctn(input_len=128, num_classes=11):
    """I/Q-only variant without SNR conditioning."""
    from model import LowSNRStem, LowSNRBlockFeat, snr_embed
    from tensorflow.keras import layers, Model, Input
    
    iq_in = Input(shape=(input_len, 2), name="iq")
    # Use a dummy zero SNR embedding
    snr_in = Input(shape=(1,), name="snr")
    cond = snr_embed(snr_in, d=32, name="cond")
    # Alternatively, set cond=None in all layers
    
    x = LowSNRStem(out_channels=64, cond_dim=32, name="stem")(iq_in, cond=None)
    x = LowSNRBlockFeat(out_channels=128, cond_dim=32, name="blk1")(x, cond=None)
    x = LowSNRBlockFeat(out_channels=128, cond_dim=32, name="blk2")(x, cond=None)
    
    gap = layers.GlobalAveragePooling1D(name="gap")(x)
    h = layers.Dense(256, activation="relu")(gap)
    h = layers.Dropout(0.2)(h)
    mod_out = layers.Dense(num_classes, activation="softmax", dtype="float32", name="mod_out")(h)
    
    return Model([iq_in, snr_in], mod_out, name="LowSNR_SCCTN_IQonly")