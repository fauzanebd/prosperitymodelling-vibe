# Database Diagram

## Overview

This document contains database diagrams for the Kesejahteraan project using Mermaid.js.

## User Model

```mermaid
erDiagram
    User {
        int id PK
        string username
        string password_hash
        boolean is_admin
    }
```

## Indicator Models

```mermaid
erDiagram
    BaseIndicator {
        int id PK
        string region
        int year
    }

    EconomicIndicator {
        int id PK
        string region
        int year
        float value
        string label_sejahtera
        string unit
    }

    HealthIndicator {
        int id PK
        string region
        int year
        float value
        string label_sejahtera
        string unit
    }

    EducationIndicator {
        int id PK
        string region
        int year
        float value
        string label_sejahtera
        string unit
    }

    InfrastructureIndicator {
        int id PK
        string region
        int year
        float value
        string label_sejahtera
        string unit
    }

    BaseIndicator ||--o{ EconomicIndicator : "extends"
    BaseIndicator ||--o{ HealthIndicator : "extends"
    BaseIndicator ||--o{ EducationIndicator : "extends"
    BaseIndicator ||--o{ InfrastructureIndicator : "extends"
```

## Economic Indicators Detail

```mermaid
erDiagram
    BaseIndicator {
        int id PK
        string region
        int year
    }

    IndeksPembangunanManusia {
        int id PK
        string region
        int year
        float value
        string label_sejahtera
        string unit
    }

    TingkatPengangguranTerbuka {
        int id PK
        string region
        int year
        float value
        string label_sejahtera
        string unit
    }

    PdrbHargaKonstan {
        int id PK
        string region
        int year
        float value
        string label_sejahtera
        string unit
    }

    PendudukMiskin {
        int id PK
        string region
        int year
        float value
        string label_sejahtera
        string unit
    }

    BaseIndicator ||--o{ IndeksPembangunanManusia : "extends"
    BaseIndicator ||--o{ TingkatPengangguranTerbuka : "extends"
    BaseIndicator ||--o{ PdrbHargaKonstan : "extends"
    BaseIndicator ||--o{ PendudukMiskin : "extends"
```

## ML Models and Predictions

```mermaid
erDiagram
    TrainedModel {
        int id PK
        string model_type
        datetime created_at
        binary model_binary
        binary scaler_binary
        text feature_names
        float accuracy
        float precision
        float recall
        float f1_score
        float test_accuracy
        float training_time
        float inference_time
        text confusion_matrix
        text feature_importance
        text cv_scores
        float mean_cv_accuracy
        float std_cv_accuracy
        text training_parameters
    }

    RegionPrediction {
        int id PK
        string region
        int year
        int model_id FK
        string predicted_class
        float prediction_probability
        datetime created_at
    }

    LabelingThreshold {
        int id PK
        string indicator
        string sejahtera_threshold
        string menengah_threshold
        string tidak_sejahtera_threshold
        string labeling_method
        boolean is_reverse
        float low_threshold
        float high_threshold
    }

    TrainedModel ||--o{ RegionPrediction : "generates"
```

## Complete Database Schema

```mermaid
erDiagram
    User {
        int id PK
        string username
        string password_hash
        boolean is_admin
    }

    BaseIndicator {
        int id PK
        string region
        int year
    }

    IndeksPembangunanManusia {
        int id PK
        string region
        int year
        float value
        string label_sejahtera
        string unit
    }

    TingkatPengangguranTerbuka {
        int id PK
        string region
        int year
        float value
        string label_sejahtera
        string unit
    }

    TrainedModel {
        int id PK
        string model_type
        datetime created_at
        binary model_binary
        binary scaler_binary
        text feature_names
        float accuracy
        float precision
        float recall
        float f1_score
        text training_parameters
    }

    RegionPrediction {
        int id PK
        string region
        int year
        int model_id FK
        string predicted_class
        float prediction_probability
        datetime created_at
    }

    LabelingThreshold {
        int id PK
        string indicator
        string sejahtera_threshold
        string menengah_threshold
        string tidak_sejahtera_threshold
        string labeling_method
        boolean is_reverse
        float low_threshold
        float high_threshold
    }

    BaseIndicator ||--o{ IndeksPembangunanManusia : "extends"
    BaseIndicator ||--o{ TingkatPengangguranTerbuka : "extends"
    TrainedModel ||--o{ RegionPrediction : "generates"
```
