import torch
import torch.nn as nn
import torchvision.models as models

from dataset import CsiroDataset, Csiro
from torchvision import transforms



class ImageEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        backbone = models.resnet18(weights="IMAGENET1K_V1")
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x



class AuxEncoder(nn.Module):
    def __init__(self, num_states, num_species, out_dim=128):
        super().__init__()

        self.state_emb = nn.Embedding(num_states, 8)       #  check
        self.species_emb = nn.Embedding(num_species, 16)   #  check

        self.mlp = nn.Sequential(
            nn.Linear(2 + 8 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, extras, state, species):
        s = self.state_emb(state)
        sp = self.species_emb(species)

        x = torch.cat([extras, s, sp], dim=1)
        return self.mlp(x)
    


class BiomassModel(nn.Module):
    def __init__(
        self,
        num_states,
        num_species,
        num_targets=5
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(out_dim=512)
        self.aux_encoder = AuxEncoder(
            num_states=num_states,
            num_species=num_species,
            out_dim=128
        )

        self.head = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, num_targets)
        )

        # head مخصوص inference (image-only)
        self.image_only_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_targets)
        )

    def forward(
        self,
        image,
        extras=None,
        state=None,
        species=None,
        # train_mode=True
    ):
        img_feat = self.image_encoder(image)


        if self.training:   
            aux_feat = self.aux_encoder(extras, state, species)
            feat = torch.cat([img_feat, aux_feat], dim=1)
            return self.head(feat)
        else:
            return self.image_only_head(img_feat)



if __name__ == "__main__":

    # Example usage
    model = BiomassModel(num_states=4, num_species=15, num_targets=5)


    dummy_image = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    dummy_extras = torch.randn(4, 2)           # Batch of 4 extra features
    dummy_state = torch.randint(0, 4, (4,))   # Batch of 4 states
    dummy_species = torch.randint(0, 15, (4,)) # Batch of 4 species

    model.train()
    output_train = model(dummy_image, dummy_extras, dummy_state, dummy_species)
    print("Training output shape:", output_train.shape)

    model.eval()
    output_eval = model(dummy_image)
    print("Evaluation output shape:", output_eval.shape)


    ############################################################################

    Transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    ]) 

    train_loader = Csiro(
                        root="./csiro-biomass",
                        image_root="./csiro-biomass/",
                        transform=Transform,
                        valid_ratio=0.2,
                        seed=42,
                        mini=True   
                        )(batch_size=1)
    
    # print(train_loader.num_species, train_loader.num_states)
    sample = next(iter(train_loader))
    image, targets, extras, state, species = sample

    print(image.shape, targets.shape, extras.shape, state.shape, species.shape)

    model.train()
    output_train = model(image, extras, state, species)
    print("Training output shape:", output_train.shape)