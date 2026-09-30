python imgs_auditor_qwen.py --nn-dir nn --mode vision --model qwen3-vl:4b --limit 30
python imgs_auditor_qwen.py --nn-dir nn --mode vision --model qwen3-vl:4b

Rode python comparator_imgs.py e leia a lista.
Rode python comparator_imgs.py --swap --dry-run. Se a lista estiver certa, rode python comparator_imgs.py --swap.
Rode python comparator_imgs.py --quarantine para os OUTLIER.
Rode o comparador de novo. Pode aparecer uma TROCADA nova depois da troca, porque o padrão de cada letra mudou.
Só use --delete depois de olhar a quarentena e ter certeza. Os SEM_MAO são os candidatos mais seguros a apagar, e na dúvida prefira a quarentena.