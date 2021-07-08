import numpy as np
import torch
from scipy.integrate import solve_ivp


class SDESampling:
    def __init__(self, model, sde):
        self.model = model
        self.sde = sde

    def create_schedules(self, nb_steps):
        t_schedule = torch.arange(1, nb_steps + 1) / nb_steps
        t_schedule = (self.sde.t_max - self.sde.t_min) * \
            t_schedule + self.sde.t_min
        dt = t_schedule[1] - t_schedule[0]
        sigma_schedule = self.sde.sigma(t_schedule)
        beta_schedule = self.sde.beta(t_schedule)
        g_schedule = self.sde.g(t_schedule)

        return t_schedule[0], sigma_schedule, beta_schedule, g_schedule, dt

    def predict(
        self,
        audio,
        nb_steps
    ):

        with torch.no_grad():

            t_0, sigma, beta, g, dt = self.create_schedules(nb_steps)

            for n in range(nb_steps - 2, 0, -1):
                # begins at t = 1 (n = nb_steps - 1)
                # stops at t = 2/nb_steps (n=1)

                audio = (1 + dt * beta[n]/2) * audio - dt * (g[n])**2 / sigma[n] * \
                    self.model(audio, sigma[n])

                if n > 10:  # everytime
                    noise = torch.randn_like(audio)
                    audio += g[n] * dt**0.5 * noise

            # The noise level is now sigma(1/nb_steps) = sigma[0]
            # Jump step
            audio = (audio - sigma[0] * self.model(audio,
                                                   sigma[0])) / self.sde.mean(t_0)

        return audio


class SDESampling2:
    def __init__(self, model, sde):
        self.model = model
        self.sde = sde

    def create_schedules(self, nb_steps):
        t_schedule = torch.arange(0, nb_steps + 1) / nb_steps
        t_schedule = (self.sde.t_max - self.sde.t_min) * \
            t_schedule + self.sde.t_min
        sigma_schedule = self.sde.sigma(t_schedule)
        m_schedule = self.sde.mean(t_schedule)

        return sigma_schedule, m_schedule

    def predict(
        self,
        audio,
        nb_steps
    ):

        with torch.no_grad():

            sigma, m = self.create_schedules(nb_steps)

            for n in range(nb_steps - 1, 0, -1):
                # begins at t = 1 (n = nb_steps - 1)
                # stops at t = 2/nb_steps (n=1)

                audio = m[n-1] / m[n] * audio + (m[n] / m[n-1] * (sigma[n-1])**2 / sigma[n] - m[n-1] / m[n] * sigma[n]) * \
                    self.model(audio, sigma[n])

                if n > 0:  # everytime
                    noise = torch.randn_like(audio)
                    audio += sigma[n-1]*(1 - (sigma[n-1]*m[n] /
                                              (sigma[n]*m[n-1]))**2)**0.5 * noise

            # The noise level is now sigma(1/nb_steps) = sigma[0]
            # Jump step
            audio = (audio - sigma[0] * self.model(audio,
                                                   sigma[0])) / m[0]

        return audio


class ODESampling:
    def __init__(self, model, sde):
        self.model = model
        self.sde = sde

    def create_schedules(self, nb_steps):
        t_schedule = torch.arange(1, nb_steps + 1) / nb_steps
        t_schedule = (self.sde.t_max - self.sde.t_min) * \
            t_schedule + self.sde.t_min
        dt = t_schedule[1] - t_schedule[0]
        sigma_schedule = self.sde.sigma(t_schedule)
        beta_schedule = self.sde.beta(t_schedule)
        g_schedule = self.sde.g(t_schedule)

        return t_schedule[0], sigma_schedule, beta_schedule, g_schedule, dt

    def predict(
        self,
        audio,
        nb_steps
    ):

        with torch.no_grad():

            t_0, sigma, beta, g, dt = self.create_schedules(nb_steps)

            for n in range(nb_steps - 1, 0, -1):
                # begins at t = 1 (n = nb_steps - 1)
                # stops at t = 2/nb_steps (n=1)

                audio = (1 + dt * beta[n]/2) * audio - dt * (g[n])**2 / (2 * sigma[n]) * \
                    self.model(audio, sigma[n])

            # The noise level is now sigma(1/nb_steps) = sigma[0]
            # Jump step
            audio = (audio - sigma[0] * self.model(audio,
                                                   sigma[0])) / self.sde.mean(t_0)

        return audio


class ScipySolver:
    def __init__(self, model, sde):
        self.model = model
        self.sde = sde

    def ode_equation(self, t, y, batch_size, device):
        y = torch.from_numpy(y).to(device).type(torch.float32)
        y = y.reshape(batch_size, -1)
        t = torch.FloatTensor([t]).to(device)
        with torch.no_grad():
            ode = - self.sde.beta(t)/2 * y + (self.sde.g(t))**2/(2*self.sde.sigma(t)) * \
                self.model(y, self.sde.sigma(t))
        ode_numpy = ode.detach().cpu().numpy()
        ode_numpy = ode_numpy.reshape(-1)
        return ode_numpy

    def predict(self, audio):
        device = audio.device
        batch_size = audio.shape[0]
        audio = audio.reshape(-1).cpu().numpy()
        solution = solve_ivp(self.ode_equation, (0.999, 1e-4),
                             audio, rtol=1e-5, atol=1e-5, args=(batch_size, device))
        sol = solution.y[:, -1].reshape(batch_size, -1)
        return torch.from_numpy(sol).to(device)
        # TODO: peut-être ajouter un jump de fin ?


class ForwardScipySolver:
    def __init__(self, model, sde):
        self.model = model
        self.sde = sde

    def ode_equation(self, t, y, batch_size, device):
        y = torch.from_numpy(y).to(device).type(torch.float32)
        y = y.reshape(batch_size, -1)
        t = torch.FloatTensor([t]).to(device)
        with torch.no_grad():
            ode = - self.sde.beta(t)/2 * y + (self.sde.g(t))**2/(2*self.sde.sigma(t)) * \
                self.model(y, self.sde.sigma(t))
        ode_numpy = ode.detach().cpu().numpy()
        ode_numpy = ode_numpy.reshape(-1)
        return ode_numpy

    def predict(self, audio):
        device = audio.device
        batch_size = audio.shape[0]
        audio = audio.reshape(-1).cpu().numpy()
        solution = solve_ivp(self.ode_equation, (1e-4, 0.999),
                             audio, rtol=1e-5, atol=1e-5, args=(batch_size, device))
        sol = solution.y[:, -1].reshape(batch_size, -1)
        return torch.from_numpy(sol).to(device)
        # TODO: peut-être ajouter un jump de fin ?


class DDIMSampling:
    def __init__(self, model, sde):
        self.model = model
        self.sde = sde

    def create_schedules(self, nb_steps):
        t_schedule = torch.arange(1, nb_steps + 1) / nb_steps
        t_schedule = (self.sde.t_max - self.sde.t_min) * \
            t_schedule + self.sde.t_min
        sigma_schedule = self.sde.sigma(t_schedule)
        m_schedule = self.sde.mean(t_schedule)

        return sigma_schedule, m_schedule

    def predict(
        self,
        audio,
        nb_steps
    ):

        with torch.no_grad():

            sigma, m = self.create_schedules(nb_steps)

            for n in range(nb_steps - 1, 0, -1):
                # begins at t = 1 (n = nb_steps - 1)
                # stops at t = 2/nb_steps (n=1)

                audio = m[n-1] / m[n] * audio + \
                    (sigma[n-1] - sigma[n] * m[n-1] / m[n]) * \
                    self.model(audio, sigma[n])

            # The noise level is now sigma(1/nb_steps)
            # Jump step
            audio = (audio - sigma[0] * self.model(audio,
                                                   sigma[0])) / m[0]

        return audio


class ForwardODESampling:  # TODO: reflechir a propos du decalage du t_schedule
    def __init__(self, model, sde):
        self.model = model
        self.sde = sde

    def create_schedules(self, nb_steps):
        t_schedule = torch.arange(0, nb_steps + 1) / nb_steps
        t_schedule = (self.sde.t_max - self.sde.t_min) * \
            t_schedule + self.sde.t_min
        dt = t_schedule[1] - t_schedule[0]

        sigma_schedule = self.sde.sigma(t_schedule)
        beta_schedule = self.sde.beta(t_schedule)
        g_schedule = self.sde.g(t_schedule)

        return sigma_schedule, beta_schedule, g_schedule, dt

    def predict(
        self,
        audio,
        nb_steps
    ):

        with torch.no_grad():

            # let's noise audio at the level 1e-4
            audio = audio + 1e-4 * torch.randn_like(audio)
            sigma, beta, g, dt = self.create_schedules(nb_steps)

            for n in range(0, nb_steps):
                # begins at t = 0 (n = 0)
                # stops at t = 1 - 1/nb_steps (n=nb_steps - 1)

                audio = (1 - dt * beta[n]/2) * audio + dt * (g[n])**2 / (2 * sigma[n]) * \
                    self.model(audio, sigma[n])

        return audio


class ForwardDDIMSampling:
    def __init__(self, model, sde):
        self.model = model
        self.sde = sde

    def create_schedules(self, nb_steps):
        t_schedule = torch.arange(0, nb_steps + 1) / nb_steps
        t_schedule = (self.sde.t_max - self.sde.t_min) * \
            t_schedule + self.sde.t_min
        sigma_schedule = self.sde.sigma(t_schedule)
        m_schedule = self.sde.mean(t_schedule)

        return sigma_schedule, m_schedule

    def predict(
        self,
        audio,
        nb_steps
    ):

        with torch.no_grad():

            # let's noise audio at the level 1e-4
            audio = audio + 1e-4 * torch.randn_like(audio)

            sigma, m = self.create_schedules(nb_steps)

            for n in range(1, nb_steps + 1):
                # begins at t = 0 (n = 0)
                # stops at t = 1 - 1/nb_steps (n=nb_steps - 1)

                audio = m[n] / m[n-1] * audio + \
                    (sigma[n] - sigma[n-1] * m[n] / m[n-1]) * \
                    self.model(audio, sigma[n-1])

        return audio


class SDEInpainting:
    def __init__(self, model, sde):
        self.model = model
        self.sde = sde

    def create_schedules(self, nb_steps):
        t_schedule = torch.arange(1, nb_steps + 1) / nb_steps
        t_schedule = (self.sde.t_max - self.sde.t_min) * \
            t_schedule + self.sde.t_min
        dt = t_schedule[1] - t_schedule[0]
        sigma_schedule = self.sde.sigma(t_schedule)
        beta_schedule = self.sde.beta(t_schedule)
        g_schedule = self.sde.g(t_schedule)
        m_schedule = self.sde.mean(t_schedule)
        return sigma_schedule, beta_schedule, g_schedule, m_schedule, dt

    def inpaint(
        self,
        audio,
        start,
        end,
        nb_steps
    ):

        with torch.no_grad():

            sigma, beta, g, m, dt = self.create_schedules(nb_steps)
            audio_i = torch.randn_like(audio)
            for n in range(nb_steps - 1, 0, -1):
                # begins at t = 1 (n = nb_steps - 1)
                # stops at t = 2/nb_steps (n=1)

                audio_i = (1 + dt * beta[n]/2) * audio_i - dt * (g[n])**2 / sigma[n] * \
                    self.model(audio_i, sigma[n])

                if n > 0:  # everytime
                    noise = torch.randn_like(audio_i)
                    audio_i += g[n] * dt**0.5 * noise

                audio_i[:, end:] = m[n] * audio[:, end:] + \
                    sigma[n] * torch.randn_like(audio[:, end:])
                audio_i[:, :start] = m[n] * audio[:, :start] + \
                    sigma[n] * torch.randn_like(audio[:, :start])

            # The noise level is now sigma(1/nb_steps) = sigma[0]
            # Jump step
            audio_i = (audio_i - sigma[0] * self.model(audio_i,
                                                       sigma[0])) / m[0]

        return audio_i


class ClassMixingSDE:
    def __init__(self, model, classifier, sde):
        self.model = model
        self.classifier = classifier
        self.sde = sde

    def create_schedules(self, nb_steps):
        t_schedule = torch.arange(1, nb_steps + 1) / nb_steps
        t_schedule = (self.sde.t_max - self.sde.t_min) * \
            t_schedule + self.sde.t_min
        dt = t_schedule[1] - t_schedule[0]
        sigma_schedule = self.sde.sigma(t_schedule)
        beta_schedule = self.sde.beta(t_schedule)
        g_schedule = self.sde.g(t_schedule)

        return t_schedule[0], sigma_schedule, beta_schedule, g_schedule, dt

    def predict(
        self,
        audio,
        nb_steps,
        alpha_mix
    ):
        t_0, sigma, beta, g, dt = self.create_schedules(nb_steps)

        for n in range(nb_steps - 2, 0, -1):

            # (Alg 2)
            audio.requires_grad = True
            classifier_predict = self.classifier(audio, sigma[n])
            classifier_predict = torch.log(
                classifier_predict[:, 0] ** alpha_mix[0] * classifier_predict[:, 1] ** alpha_mix[1] * classifier_predict[:, 2] ** alpha_mix[2])

            grad_outputs = torch.ones_like(classifier_predict)
            grad_outputs.requires_grad = True
            grad_classifier = torch.autograd.grad(
                classifier_predict, audio, grad_outputs=grad_outputs, create_graph=True)[0]

            audio = (1 + dt * beta[n]/2) * audio - dt * (g[n])**2 / sigma[n] * \
                (self.model(audio, sigma[n]) - grad_classifier)

            if n > 10:
                noise = torch.randn_like(audio)
                audio = audio + beta[n]**0.5 * noise

            audio = audio.detach()
        audio = (audio - sigma[0] * self.model(audio,
                                               sigma[0])) / self.sde.mean(t_0)

        return audio


class ClassMixingSDE2:
    def __init__(self, model, classifier, sde):
        self.model = model
        self.classifier = classifier
        self.sde = sde

    def create_schedules(self, nb_steps):
        t_schedule = torch.arange(0, nb_steps + 1) / nb_steps
        t_schedule = (self.sde.t_max - self.sde.t_min) * \
            t_schedule + self.sde.t_min
        sigma_schedule = self.sde.sigma(t_schedule)
        m_schedule = self.sde.mean(t_schedule)

        return sigma_schedule, m_schedule

    def predict(
        self,
        audio,
        nb_steps,
        alpha_mix
    ):

        sigma, m = self.create_schedules(nb_steps)

        for n in range(nb_steps - 1, 0, -1):
            # begins at t = 1 (n = nb_steps - 1)
            # stops at t = 2/nb_steps (n=1)

            audio.requires_grad = True
            classifier_predict = self.classifier(audio, sigma[n])
            classifier_predict = torch.log(
                classifier_predict[:, 0] ** alpha_mix[0] * classifier_predict[:, 1] ** alpha_mix[1] * classifier_predict[:, 2] ** alpha_mix[2])

            grad_outputs = torch.ones_like(classifier_predict)
            grad_outputs.requires_grad = True
            grad_classifier = torch.autograd.grad(
                classifier_predict, audio, grad_outputs=grad_outputs, create_graph=True)[0]

            audio = m[n-1] / m[n] * audio + (m[n] / m[n-1] * (sigma[n-1])**2 / sigma[n] - m[n-1] / m[n] * sigma[n]) * \
                (self.model(audio, sigma[n]) - grad_classifier)

            if n > 0:  # everytime
                noise = torch.randn_like(audio)
                audio += sigma[n-1]*(1 - (sigma[n-1]*m[n] /
                                          (sigma[n]*m[n-1]))**2)**0.5 * noise

            audio = audio.detach()

        # The noise level is now sigma(1/nb_steps) = sigma[0]
        # Jump step
        audio = (audio - sigma[0] * self.model(audio,
                                               sigma[0])) / m[0]

        return audio


class ClassMixingODE:
    def __init__(self, model, classifier, sde):
        self.model = model
        self.classifier = classifier
        self.sde = sde

    def create_schedules(self, nb_steps):
        t_schedule = torch.arange(1, nb_steps + 1) / nb_steps
        t_schedule = (self.sde.t_max - self.sde.t_min) * \
            t_schedule + self.sde.t_min
        dt = t_schedule[1] - t_schedule[0]
        sigma_schedule = self.sde.sigma(t_schedule)
        beta_schedule = self.sde.beta(t_schedule)
        g_schedule = self.sde.g(t_schedule)

        return t_schedule[0], sigma_schedule, beta_schedule, g_schedule, dt

    def predict(
        self,
        audio,
        nb_steps,
        alpha_mix
    ):

        t_0, sigma, beta, g, dt = self.create_schedules(nb_steps)

        for n in range(nb_steps - 1, 0, -1):
            # begins at t = 1 (n = nb_steps - 1)
            # stops at t = 2/nb_steps (n=1)
            audio.requires_grad = True
            classifier_predict = self.classifier(audio, sigma[n])
            classifier_predict = torch.log(
                classifier_predict[:, 0] ** alpha_mix[0] * classifier_predict[:, 1] ** alpha_mix[1] * classifier_predict[:, 2] ** alpha_mix[2])

            grad_outputs = torch.ones_like(classifier_predict)
            grad_outputs.requires_grad = True
            grad_classifier = torch.autograd.grad(
                classifier_predict, audio, grad_outputs=grad_outputs, create_graph=True)[0]

            audio = (1 + dt * beta[n]/2) * audio - dt * (g[n])**2 / (2 * sigma[n]) * \
                (self.model(audio, sigma[n]) - grad_classifier)

            audio = audio.detach()

        # The noise level is now sigma(1/nb_steps) = sigma[0]
        # Jump step
        audio = (audio - sigma[0] * self.model(audio,
                                               sigma[0])) / self.sde.mean(t_0)

        return audio


class ClassMixingDDIM:
    def __init__(self, model, classifier, sde):
        self.model = model
        self.classifier = classifier
        self.sde = sde

    def create_schedules(self, nb_steps):
        t_schedule = torch.arange(1, nb_steps + 1) / nb_steps
        t_schedule = (self.sde.t_max - self.sde.t_min) * \
            t_schedule + self.sde.t_min
        sigma_schedule = self.sde.sigma(t_schedule)
        m_schedule = self.sde.mean(t_schedule)

        return sigma_schedule, m_schedule

    def predict(
        self,
        audio,
        nb_steps,
        alpha_mix
    ):
        sigma, m = self.create_schedules(nb_steps)

        for n in range(nb_steps - 1, 0, -1):
            # begins at t = 1 (n = nb_steps - 1)
            # stops at t = 2/nb_steps (n=1)
            audio.requires_grad = True
            classifier_predict = self.classifier(audio, sigma[n])
            classifier_predict = torch.log(
                classifier_predict[:, 0] ** alpha_mix[0] * classifier_predict[:, 1] ** alpha_mix[1] * classifier_predict[:, 2] ** alpha_mix[2])

            grad_outputs = torch.ones_like(classifier_predict)
            grad_outputs.requires_grad = True
            grad_classifier = torch.autograd.grad(
                classifier_predict, audio, grad_outputs=grad_outputs, create_graph=True)[0]

            audio = m[n-1] / m[n] * audio + \
                (sigma[n-1] - sigma[n] * m[n-1] / m[n]) * \
                (self.model(audio, sigma[n]) - grad_classifier)

            audio = audio.detach()

        # The noise level is now sigma(1/nb_steps)
        # Jump step
        audio = (audio - sigma[0] * self.model(audio,
                                               sigma[0])) / m[0]

        return audio
