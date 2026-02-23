            tilt = 1.0
            face = 4
            alpha_tilt = tilt*(jnp.pi / 4 )
            print(f"Test case 2: (alpha_tilt = {alpha_tilt}, {alpha_tilt}, 0)")
            X0, Y0, Z0 = equiangular_to_cartesian(
                jnp.array(alpha_tilt), jnp.array(0.0), face)
            #    jnp.array(alpha_tilt), jnp.array(alpha_tilt), face)
            X0, Y0, Z0 = float(X0), float(Y0), float(Z0)
