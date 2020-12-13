
def convert_tensor2array(tensor):
        '''Função para converter de tensor para array
            Parâmetros:
                tensor: tensor
        '''
        return tensor.cpu().detach().numpy()[0]