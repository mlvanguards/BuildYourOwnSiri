plugins {
    id("com.android.asset-pack")
}

assetPack {
    // this must exactly match the folder name of your model_pack directory
    packName.set("model_pack")
    dynamicDelivery {
        // choose one delivery mode:
        deliveryType.set("install-time")  // or "fast-follow" or "on-demand"
    }
}
