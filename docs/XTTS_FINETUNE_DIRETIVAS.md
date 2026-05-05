# Diretivas obrigatorias de treino (XTTS)

Estas diretivas valem para **cada experimento de treino** executado neste projeto.

## Artefatos obrigatorios por experimento

1. Checkpoint/adaptador final apos o fine-tune.
2. Tempo medio por epoca.
3. Tempo medio por step.
4. Grafico de perda por epoca.
5. Grafico de perda por step.
6. Inferencia para cada checkpoint salvo durante o treino.

## Logs obrigatorios em `train.log`

Cada execucao de treino deve registrar no minimo:

1. Identificacao do experimento (`run_name`, checkpoint base, modelo base).
2. Hiperparametros principais (epocas/steps, LR, batch, grad accumulation, max seq len).
3. Configuracao XTTS ativa (device, sample rate, parametros de inferencia/treino relevantes).
4. Estatisticas de dataset (total lido, total aproveitado, total descartado e motivo).
5. Progresso do treino e avaliacoes (logs do Trainer com `loss` / `eval_loss` quando houver).
6. Checklist final dos artefatos esperados (presente/ausente).

## Regra de emissao de logs

- Todo log relevante deve aparecer **na tela** e tambem ser persistido em **arquivo** (`train.log`).
- Nao manter logs importantes apenas no terminal ou apenas em arquivo.

## Mapeamento de saidas no projeto

- `ckpts/<dataset>/model_last.pt`: checkpoint final.
- `ckpts/<dataset>/experiment_report/epoch_timing.csv`: tempos por epoca (base para media).
- `ckpts/<dataset>/experiment_report/step_timing.csv`: tempos por step (base para media).
- `ckpts/<dataset>/graphics/loss_by_epoch.png` e/ou `ckpts/<dataset>/experiment_report/loss_by_epoch.png`.
- `ckpts/<dataset>/graphics/loss.png` e/ou `ckpts/<dataset>/experiment_report/loss.png`.
- `ckpts/<dataset>/samples/step_*_gen.wav` e `step_*_ref.wav`: inferencias por checkpoints/saves.

## Uso recomendado neste repositorio (XTTS PT-BR)

- Dataset filtrado: `data/filtered`
- Launcher recomendado para 3 epocas: `run-xtts-ptbr-filtered-3ep.bat`
- Log da execucao: `data/xtts-experiments/logs/xtts-ptbr-<timestamp>.log`
- Artefatos por experimento: `data/xtts-experiments/ptbr-<timestamp>/artifacts`
