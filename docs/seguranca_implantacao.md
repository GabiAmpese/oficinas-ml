# Segurança, privacidade e entrega — limites do protótipo

## Estado atual
O trabalho utiliza dados históricos reais, identificados por códigos de veículos e fontes. Isso é **pseudonimização**, e não garante anonimização irreversível. O Streamlit de demonstração carrega arquivos locais e um artefato `joblib`; não há comprovação de autenticação, autorização por oficina ou monitoramento de produção.

## Cuidados antes de uma implantação real
1. **Revisão de dados:** verificar ausência de placas, nomes, telefones, CPF e campos livres identificadores, inclusive metadados das planilhas, logs, commits e histórico do Git.
2. **Controle de acesso:** separar usuários por oficina, autenticar e autorizar consultas no servidor; filtros visuais não são mecanismo de autorização.
3. **Dados e retenção:** definir responsáveis, finalidade, base legal adequada, regras de retenção e exclusão; revisar acesso aos backups.
4. **Modelo:** carregar apenas artefatos `joblib` de origem confiável (a desserialização pode executar código); registrar versão e treinamento correspondente.
5. **Operação:** registrar erros e versões sem gravar dados pessoais nos logs; monitorar falhas de leitura de dados, erros de previsão e distribuição do uso.
6. **Recuperação:** manter cópias íntegras dos artefatos e dados autorizados, testar restauração e documentar procedimento de rollback.
7. **Deploy:** publicar somente depois de validar permissões, dados expostos, configuração do ambiente, endpoint e comportamento da aplicação.

## Decisões fora do escopo acadêmico
Não criar envio automático de WhatsApp, banco de clientes, servidor MCP, agentes que contatam clientes ou acesso público ao painel apenas para demonstrar mais tecnologias. Essas extensões exigem autorização, projeto de segurança e novos testes.

## Checklist do quality gate
- [ ] Dependências instaladas em ambiente isolado
- [ ] Preparação e treinamento reproduzidos localmente
- [ ] Testes automatizados executados e resultado registrado
- [ ] Cenários de aceite conferidos na interface
- [ ] Nenhum dado identificador exposto no repositório ou no deploy
- [ ] Modelo e métricas correspondem à mesma execução
- [ ] Limitações e pendências registradas no README/relatório
