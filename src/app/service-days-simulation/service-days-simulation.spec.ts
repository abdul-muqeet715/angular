import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ServiceDaysSimulation } from './service-days-simulation';

describe('ServiceDaysSimulation', () => {
  let component: ServiceDaysSimulation;
  let fixture: ComponentFixture<ServiceDaysSimulation>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ServiceDaysSimulation]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ServiceDaysSimulation);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
