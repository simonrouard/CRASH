from sde import SDE, VpSdeSigmoid, VpSdeCos, GeneralizedSubVpSdeCos, SubVpSdeCos


def get_sde(sde_type: str,
            sde_kwargs,) -> SDE:

    sigma_min = sde_kwargs['sigma_min'] if 'sigma_min' in sde_kwargs else None
    sigma_max = sde_kwargs['sigma_max'] if 'sigma_max' in sde_kwargs else None
    gamma = sde_kwargs['gamma'] if 'gamma' in sde_kwargs else None
    eta = sde_kwargs['eta'] if 'eta' in sde_kwargs else None
    # mle_training = sde_kwargs['mle_training']

    if sde_type == 'vp-sigmoid':
        # return VpSdeSigmoid(mle_training)
        return VpSdeSigmoid()

    if sde_type == 'vp-cos':
        assert ((sigma_min is not None) and (sigma_max is not None))
        # return VpSdeCos(mle_training, sigma_min, sigma_max)
        return VpSdeCos(sigma_min, sigma_max)

    if sde_type == 'subvp-cos':
        assert ((sigma_min is not None) and (sigma_max is not None))
        # return SubVpSdeCos(mle_training, sigma_min, sigma_max)
        return SubVpSdeCos(sigma_min, sigma_max)

    if sde_type == 'generalized-sub-vp-cos':
        assert ((sigma_min is not None) and (sigma_max is not None))
        assert ((gamma is not None) and (eta is not None))
        # return GeneralizedSubVpSdeCos(mle_training, gamma, eta, sigma_min,
        #                               sigma_max)
        return GeneralizedSubVpSdeCos(gamma, eta, sigma_min,
                                      sigma_max)

    else:
        raise NotImplementedError
