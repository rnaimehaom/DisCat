# DisCat
A Variational Autoencoder paired with a Quantitative Structure Activity Relationship model to generate catalytic molecules while 
optimizing the binding energy value for the artificial nitrogen fixation reaction

![Capture](https://user-images.githubusercontent.com/66485156/112565460-15d2e980-8dab-11eb-80a8-a2e4fb4cf8df.PNG)

## How It Works

### Molecular Generation VAE
The variational autoencoder was trained on the ZINC15 dataset of molecules in which molecule was paired with its respective binding energy 
with nitrogen gas (this chemical property determines the catalytic activity of a molecule for nitrogen fixation). The model was built with 
three section: the encoder, quantitative structure activity layers, and a decoder. The encoder encodes SMILES molecules into a latent space
representation, which is enhanced to determine catalytic molecules with an optimized binding energy value. The decoder then recreates the molecule
from the latent space representation. The model can be found in [VAE_models.py](https://github.com/roshanmehta/DisCat/blob/master/VAE_files/VAE_models.py).

### Web Application
A web application was developed to create a simple and easily accessible application for the models for future researchers and companies.
The web server consists of a text box ,where the user can input their desired binding energy value, and a button that will trigger the 
value to go through the model and generate a molecule in SMILES representation. However, the server is yet to be deployed on the internet for
public use.

![Picture1](https://user-images.githubusercontent.com/66485156/112567796-706e4480-8daf-11eb-8cb4-2210534c9dd2.png)  ![Capture](https://user-images.githubusercontent.com/66485156/112567825-77955280-8daf-11eb-96be-d0a78fbbfe7d.PNG)

## References
* Major reference and code from [Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules](https://github.com/aspuru-guzik-group/chemical_vae)
